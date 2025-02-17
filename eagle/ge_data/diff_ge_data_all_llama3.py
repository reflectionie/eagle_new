import os
import torch
import argparse
import safetensors
from tqdm import tqdm

# 用于遍历文件夹，列出所有 ckpt 文件
def list_files(path):
    ckpt_paths = []
    for root, directories, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                file_path = os.path.join(root, file)
                ckpt_paths.append(file_path)
    return ckpt_paths

# --------------------- 参数解析 ---------------------
parser = argparse.ArgumentParser(description="Process ckpt files using ea_model to generate draft_hidden.")
parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the ckpt directory")
parser.add_argument("--ckpt_path_ea_model", type=str, required=True, help="Path to the ea_model directory")
parser.add_argument("--gpu_index", type=int, default=0, help="GPU index to use")
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
ckpt_path_ea_model = args.ckpt_path_ea_model
gpu_index = args.gpu_index

# --------------------- 设置设备 ---------------------
device = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu':
    torch.cuda.set_device(device)
print(f"Using device: {device}")

# --------------------- 加载 ea_model ---------------------
# 按照你的方式加载模型
from ..model.configs import EConfig
from ..model.cnets import Model

# 加载配置和模型
config = EConfig.from_pretrained(ckpt_path_ea_model)
model = Model(config).to(device).eval()

# 加载 ea_model 权重
ea_model_path = ckpt_path_ea_model
load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
if os.path.exists(load_model_path):
    ea_layer_state_dict = torch.load(load_model_path, map_location=device)
else:
    load_model_path = os.path.join(ea_model_path, "model.safetensors")
    ea_layer_state_dict = safetensors.torch.load_file(load_model_path, device=device)

model.load_state_dict(ea_layer_state_dict, strict=True)
print(f"Loaded ea_model from {load_model_path}")

# --------------------- 处理 ckpt 文件 ---------------------
ckpt_files = list_files(ckpt_dir)
print(f"Found {len(ckpt_files)} ckpt files in {ckpt_dir}")

pad_token_id = 0  # 如有需要，可根据实际情况设置

model.half()

for ckpt_path in tqdm(ckpt_files, desc="Processing ckpt files"):
    data = torch.load(ckpt_path, map_location="cpu")
    # 假设 data 至少包含以下字段：
    #   "input_ids": [T] 或 [1, T]
    #   "hidden_state": [T, hidden_dim] 或 [1, T, hidden_dim]
    
    # 保证 input_ids 有 batch 维度
    input_ids = data["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, T]
    
    gt_hidden = data["hidden_state"]
    if gt_hidden.dim() == 2:
        gt_hidden = gt_hidden.unsqueeze(0)  # [1, T, hidden_dim]
    
    input_ids = input_ids.to(device)
    gt_hidden = gt_hidden.to(device)

    # 如果有 loss_mask 等其他张量，也请一并处理，这里仅示例 input_ids 与 hidden_state

    # --------------------- 截断对齐 ---------------------
    # 按要求：input_ids 去掉第0个 token，hidden_state 去掉最后一个 token
    bsz, T = input_ids.shape  # 原始长度 T
    _, _, hidden_dim = gt_hidden.shape

    shifted_input_ids = input_ids[:, 1:]          # [bsz, T-1]
    shifted_gt_hidden = gt_hidden[:, :-1, :]         # [bsz, T-1, hidden_dim]

    # 若有 loss_mask 或 attention_mask，可同步截断，如：
    shifted_attention_mask = (shifted_input_ids != pad_token_id).long().to(device)

    # --------------------- 使用 ea_model生成 draft_hidden ---------------------
    # 假设模型前向接口为:
    #   model(hidden_states, input_ids, attention_mask)
    # 模型输出 draft_hidden 的 shape 为 [bsz, (T-1)+1, hidden_dim]（多预测一个 token）
    with torch.no_grad():
        draft_hidden = model(
            hidden_states=shifted_gt_hidden,
            input_ids=shifted_input_ids,
            attention_mask=shifted_attention_mask
        )
        # 如果 draft_hidden 多出一个 token（即 shape 为 [bsz, T, hidden_dim]），截断最后一个 token
        if draft_hidden.shape[1] == (T - 1):
            draft_hidden = draft_hidden[:, :-1, :]

    # --------------------- 写回数据 ---------------------
    # 更新 data 中的字段，确保 input_ids, hidden_state, draft_hidden 长度一致（均为 T-1）
    data["input_ids"] = shifted_input_ids[:, 1:].cpu()
    data["hidden_state"] = shifted_gt_hidden[:, 1:, :].cpu()
    data["draft_hidden"] = draft_hidden.cpu()
    # 如有 loss_mask 等其他数据，也请同步更新

    torch.save(data, ckpt_path)
