#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader

# 导入模型相关代码
from eagle.model.ea_model_rl import EaModel
# 如果需要可以使用 fastchat 中的工具转换 dtype（也可以直接用 torch.float16）
from fastchat.utils import str_to_torch_dtype

def list_files(path):
    files = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            files.append(os.path.join(root, f))
    return files

class CustomDataset(Dataset):
    def __init__(self, datapath):
        self.files = list_files(datapath)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        data = torch.load(self.files[index])
        # 同时返回 input_ids 与 loss_mask
        return {
            "input_ids": data["input_ids"],
            "loss_mask": data["loss_mask"]
        }

def get_allowed_prefix_lengths(loss_mask):
    """
    对 loss_mask（1D 张量）分段，返回允许的前缀长度列表。
    对于连续的 mask==0 区段，只允许取整个区段；
    对于 mask==1 区段，允许取从区段起始（累计）后任意部分。
    """
    mask_list = loss_mask.tolist()
    segments = []
    current_val = mask_list[0]
    start = 0
    for i, val in enumerate(mask_list):
        if val != current_val:
            segments.append((start, i - 1, current_val))
            start = i
            current_val = val
    segments.append((start, len(mask_list) - 1, current_val))
    
    allowed = []
    cumulative = 0
    for (s, e, val) in segments:
        seg_len = e - s + 1
        if val == 0:
            cumulative += seg_len
            # 对于 mask==0 的区段，只允许保留整个区段
            allowed.append(cumulative)
        else:
            start_cum = cumulative
            cumulative += seg_len
            # 对于 mask==1 的区段，允许从 1 到 seg_len 个 token 的前缀
            allowed.extend(list(range(start_cum + 1, cumulative + 1)))
    return allowed

def main():
    parser = argparse.ArgumentParser(description="使用训练数据集进行推理，并打印结果（对每个样本每个前缀都推理一次）")
    parser.add_argument("--base-model-path", type=str, default="/net/papilio/storage7/tingyuan/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct",
                        help="基础模型所在目录")
    parser.add_argument("--ea_model_path", type=str, default="/net/graphium/storage3/tingyuan/models/EAGLE-LLaMA3-Instruct-8B",
                        help="EAGLE权重所在目录或仓库")
    parser.add_argument("--datapath", type=str, default="/net/papilio/storage7/tingyuan/llama/eagle_new/eagle/reflectio/train_data/eagle_data/sharegpt_0_300_mufp16",
                        help="训练数据集目录（存有 torch.save 文件的目录）")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大新 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度，0表示贪婪")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备，例如 'cuda' 或 'cpu'")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float64", "float16", "bfloat16"],
                        help="模型数据类型")
    parser.add_argument("--start_idx", type=int, default=50, help="start to split prefix")
    args = parser.parse_args()

    # 加载预训练模型
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=60,
        depth=5,
        top_k=10,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    model.to(args.device)
    model.eval()

    # 构造数据集与 DataLoader（每个样本 batch size = 1）
    dataset = CustomDataset(args.datapath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in dataloader:
        input_ids = sample["input_ids"].to(args.device)
        loss_mask = sample["loss_mask"].to(args.device).squeeze()  # 假设形状为 (seq_len,)
        full_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        print("原始 Prompt:", full_text)

        # 根据 loss_mask 计算允许的前缀长度
        allowed_prefixes = get_allowed_prefix_lengths(loss_mask)
        print("允许的前缀长度列表：", len(allowed_prefixes))

        # 对每个允许的前缀长度进行生成
        for p_len in allowed_prefixes:
            prefix = input_ids[:, :p_len]
            # 每次调用 eagenerate 时内部会重置 kv_cache
            generated, new_token, step = model.eagenerate(
                prefix, 
                temperature=args.temperature, 
                max_new_tokens=args.max_new_tokens, 
                log=True
            )
            p_len = prefix.shape[1]
            generated_tokens = generated[0][p_len:]
            output_text = tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=True)
            print(f"前缀长度 {p_len}，新增 token: {output_text}")

if __name__ == "__main__":
    main()
