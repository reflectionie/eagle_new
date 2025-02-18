import os
import torch
import argparse
import safetensors
from tqdm import tqdm

def list_files(path):
    ckpt_paths = []
    for root, directories, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_paths.append(os.path.join(root, file))
    return sorted(ckpt_paths)  # 加 sorted，方便按照固定顺序处理

parser = argparse.ArgumentParser(description="Process ckpt files using ea_model to generate draft_hidden.")
parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the ckpt directory")
parser.add_argument("--ckpt_path_ea_model", type=str, required=True, help="Path to the ea_model directory")
parser.add_argument("--gpu_index", type=int, default=0, help="GPU index to use")
# 新增的参数，用于指定只处理区间 [start_idx, end_idx)
parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing ckpt files")
parser.add_argument("--end_idx", type=int, default=-1, help="End index (exclusive). If -1, process until the end.")
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
ckpt_path_ea_model = args.ckpt_path_ea_model
gpu_index = args.gpu_index
start_idx = args.start_idx
end_idx = args.end_idx

device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
if "cuda" in device:
    torch.cuda.set_device(device)
print(f"Using device: {device}")

# --------------------- 加载 ea_model ---------------------
from ..model.configs import EConfig
from ..model.cnets import Model

config = EConfig.from_pretrained(ckpt_path_ea_model)
model = Model(config).to(device).eval()

# 尝试加载 pytorch_model.bin，否则加载 model.safetensors
bin_path = os.path.join(ckpt_path_ea_model, "pytorch_model.bin")
if os.path.exists(bin_path):
    ea_layer_state_dict = torch.load(bin_path, map_location=device)
else:
    st_path = os.path.join(ckpt_path_ea_model, "model.safetensors")
    ea_layer_state_dict = safetensors.torch.load_file(st_path, device=device)

model.load_state_dict(ea_layer_state_dict, strict=True)
print(f"Loaded ea_model from {ckpt_path_ea_model}")

model.half()

# --------------------- 获取所有 .ckpt 文件并根据索引截取 ---------------------
ckpt_files = list_files(ckpt_dir)
total_files = len(ckpt_files)
print(f"Found {total_files} ckpt files in {ckpt_dir}")

# 如果 end_idx 为 -1 或大于文件总数，就处理到最后
if end_idx == -1 or end_idx > total_files:
    end_idx = total_files

# 只处理 [start_idx, end_idx) 范围的文件
ckpt_files_slice = ckpt_files[start_idx:end_idx]
print(f"Will process files in range: [{start_idx}, {end_idx}), total = {len(ckpt_files_slice)}")

pad_token_id = 0  # 如有需要，可根据你的实际情况调整

# --------------------- 创建新文件夹 ---------------------
ckpt_dir_abs = os.path.abspath(ckpt_dir)
parent_dir = os.path.dirname(ckpt_dir_abs)
orig_folder_name = os.path.basename(ckpt_dir_abs)
new_folder_name = f"draft_{orig_folder_name}"
new_dir = os.path.join(parent_dir, new_folder_name)
os.makedirs(new_dir, exist_ok=True)
print(f"New ckpt files will be saved to: {new_dir}")

# --------------------- 逐个处理 .ckpt 文件 ---------------------
for ckpt_path in tqdm(ckpt_files_slice, desc="Processing ckpt files"):
    # 1) 读取原数据
    data = torch.load(ckpt_path, map_location="cpu")

    # 假设 data 至少包含：
    #   "input_ids": [T] 或 [1, T]
    #   "hidden_state": [T, hidden_dim] 或 [1, T, hidden_dim]
    input_ids = data["input_ids"]
    gt_hidden = data["hidden_state"]

    # 保证有 batch 维度
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # => [bsz, T]
    if gt_hidden.dim() == 2:
        gt_hidden = gt_hidden.unsqueeze(0)  # => [bsz, T, hidden_dim]

    # -- loss_mask 相关处理：先判断有没有 loss_mask --
    has_loss_mask = ("loss_mask" in data)
    if has_loss_mask:
        loss_mask = data["loss_mask"]
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(0)  # => [bsz, T]
        loss_mask = loss_mask.to(device)
    else:
        loss_mask = None

    # 放到 GPU / CPU
    input_ids = input_ids.to(device)
    gt_hidden = gt_hidden.to(device)

    # --------------------- 截断对齐（第一段） ---------------------
    bsz, T = input_ids.shape
    _, _, hidden_dim = gt_hidden.shape

    # 去掉 input_ids 的第 0 个 token
    shifted_input_ids = input_ids[:, 1:]         # => [bsz, T-1]
    # 去掉 hidden_state 的最后 1 个 token
    shifted_gt_hidden = gt_hidden[:, :-1, :]     # => [bsz, T-1, hidden_dim]

    # 同步截断 loss_mask（如果有）
    if loss_mask is not None:
        shifted_loss_mask = loss_mask[:, :-1]    # => [bsz, T-1]
    else:
        shifted_loss_mask = None

    # 计算 attention_mask
    shifted_attention_mask = (shifted_input_ids != pad_token_id).long().to(device)

    # --------------------- 使用 ea_model 生成 draft_hidden ---------------------
    with torch.no_grad():
        draft_hidden = model(
            hidden_states=shifted_gt_hidden,
            input_ids=shifted_input_ids,
            attention_mask=shifted_attention_mask
        )
        # 这里如果 draft_hidden 比较特殊，需要再做一次截断
        # 例如你说要去掉最后 1 个 token，可以再加一行：
        if draft_hidden.shape[1] == (T - 1):
            draft_hidden = draft_hidden[:, :-1, :]

    # --------------------- 写回 data （第二段截断） ---------------------
    final_input_ids = shifted_input_ids[:, 1:].cpu()         # => [bsz, T-2]
    final_hidden_state = shifted_gt_hidden[:, 1:, :].cpu()   # => [bsz, T-2, hidden_dim]

    if shifted_loss_mask is not None:
        final_loss_mask = shifted_loss_mask[:, 1:].cpu()     # => [bsz, T-2]
    else:
        final_loss_mask = None

    data["input_ids"] = final_input_ids
    data["hidden_state"] = final_hidden_state
    data["draft_hidden"] = draft_hidden.cpu()  # => [bsz, T-2, hidden_dim]
    if final_loss_mask is not None:
        data["loss_mask"] = final_loss_mask    # 保存

    # --------------------- 保存到新文件 ---------------------
    orig_filename = os.path.basename(ckpt_path)
    new_filename = orig_filename
    new_ckpt_path = os.path.join(new_dir, new_filename)
    torch.save(data, new_ckpt_path)

print("Done! Your new .ckpt files have been saved to the 'draft_*' folder.")
