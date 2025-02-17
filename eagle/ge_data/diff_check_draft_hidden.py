#!/usr/bin/env python
import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoConfig
import safetensors
from safetensors.torch import safe_open

# 初始化 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------- 参数解析 ---------------------
parser = argparse.ArgumentParser(description="Evaluate draft_hidden vs gt_hidden using LM Head")
parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing ckpt files")
parser.add_argument("--basepath", type=str, default="/net/graphium/storage3/tingyuan/models/Meta-Llama-3-8B-Instruct/",
                    help="Path to LM Head base model")
parser.add_argument("--gpu_index", type=int, default=0, help="GPU index to use")
args = parser.parse_args()

# --------------------- 设备设置 ---------------------
device = f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu':
    torch.cuda.set_device(device)
logger.info(f"Using device: {device}")

# --------------------- 加载 LM Head ---------------------
head = None
if args.basepath is not None:
    logger.info(f"Loading LM Head from: {args.basepath}")
    baseconfig = AutoConfig.from_pretrained(args.basepath)
    # 构造一个线性层：输入 hidden_size，输出 vocab_size，无 bias
    head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
    
    try:
        index_file = os.path.join(args.basepath, "model.safetensors.index.json")
        with open(index_file, "r") as f:
            index_json = json.load(f)
        head_path = index_json["weight_map"]["lm_head.weight"]
        safe_head_path = os.path.join(args.basepath, head_path)
        with safe_open(safe_head_path, framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            # 确保切片到 hidden_dim 大小
            tensor = tensor_slice[:, :hidden_dim].float()
    except Exception as e:
        logger.info(f"Safe tensor loading failed with {e}, try pytorch_model.bin")
        index_file = os.path.join(args.basepath, "pytorch_model.bin.index.json")
        with open(index_file, "r") as f:
            index_json = json.load(f)
        head_path = index_json["weight_map"]["lm_head.weight"]
        weight_file = os.path.join(args.basepath, head_path)
        weights = torch.load(weight_file, map_location="cpu")
        tensor = weights["lm_head.weight"].float()
    
    head.weight.data = tensor
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    head = head.to(device)
    logger.info("LM Head loaded and moved to device.")
else:
    logger.info("LM Head evaluation disabled as no basepath provided.")

# --------------------- 遍历 ckpt 文件 ---------------------
def list_ckpt_files(path):
    ckpt_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_paths.append(os.path.join(root, file))
    return ckpt_paths

ckpt_files = list_ckpt_files(args.ckpt_dir)
logger.info(f"Found {len(ckpt_files)} ckpt files in {args.ckpt_dir}")

# --------------------- 评估指标统计 ---------------------
top1_in_top1_sum = 0
top1_in_top2_sum = 0
top1_in_top3_sum = 0
total_tokens = 0  # 总共评估的 token 数

# --------------------- 评估过程 ---------------------
# 假设每个 ckpt 文件中均包含：
#    "hidden_state": ground truth hidden state, shape: [B, T, hidden_dim]
#    "draft_hidden":  draft hidden state, shape: [B, T, hidden_dim]
# 这里通常 B 为 1，但代码支持 batch 维度

if head is not None:
    for ckpt_path in tqdm(ckpt_files, desc="Evaluating ckpt files"):
        data = torch.load(ckpt_path, map_location="cpu")
        # 加载 ground truth 和 draft 的 hidden state
        gt_hidden = data["hidden_state"]
        draft_hidden = data["draft_hidden"]

        if gt_hidden.dim() == 2:
            gt_hidden = gt_hidden.unsqueeze(0)
        if draft_hidden.dim() == 2:
            draft_hidden = draft_hidden.unsqueeze(0)
        
        gt_hidden = gt_hidden.to(device)
        draft_hidden = draft_hidden.to(device)
        
        # 展开为 [B*T, hidden_dim]
        B, T, hidden_dim = gt_hidden.shape
        total_tokens += B * T
        
        gt_hidden_flat = gt_hidden.reshape(-1, hidden_dim)
        draft_hidden_flat = draft_hidden.reshape(-1, hidden_dim)
        
        # 注意：原始数据为 float16，而 LM Head 的权重为 float32
        # 这里将输入转换为 float32 保证数据类型一致
        gt_hidden_flat = gt_hidden_flat.float()
        draft_hidden_flat = draft_hidden_flat.float()
        
        # 使用 LM Head 得到 logits，shape: [B*T, vocab_size]
        gt_logits = head(gt_hidden_flat)
        draft_logits = head(draft_hidden_flat)
        
        # 取 top-3 token id，shape: [B*T, 3]
        k = 3
        gt_topk = torch.topk(gt_logits, k=k, dim=-1).indices
        draft_topk = torch.topk(draft_logits, k=k, dim=-1).indices
        
        # 取 top1 预测
        draft_top1 = draft_topk[:, 0]  # [B*T]
        gt_top1 = gt_topk[:, 0]        # [B*T]
        # 同时取前2个作为 top2 用于后续判断
        gt_top2 = gt_topk[:, :2]       # [B*T, 2]
        
        # 累计统计每个 token 的匹配情况
        for i in range(draft_top1.size(0)):
            pred_token = draft_top1[i].item()
            gt_token = gt_top1[i].item()
            gt_top2_tokens = gt_top2[i].tolist()
            gt_top3_tokens = gt_topk[i].tolist()
            
            if pred_token == gt_token:
                top1_in_top1_sum += 1
            if pred_token in gt_top2_tokens:
                top1_in_top2_sum += 1
            if pred_token in gt_top3_tokens:
                top1_in_top3_sum += 1

    # 计算各指标
    top1_in_top1_prob = top1_in_top1_sum / total_tokens if total_tokens > 0 else 0.0
    top1_in_top2_prob = top1_in_top2_sum / total_tokens if total_tokens > 0 else 0.0
    top1_in_top3_prob = top1_in_top3_sum / total_tokens if total_tokens > 0 else 0.0

    logger.info(f"Total tokens evaluated: {total_tokens}")
    logger.info(f"Top1 accuracy (exact): {top1_in_top1_prob:.4f}")
    logger.info(f"Top1 in top2 accuracy: {top1_in_top2_prob:.4f}")
    logger.info(f"Top1 in top3 accuracy: {top1_in_top3_prob:.4f}")
else:
    logger.info("LM Head evaluation is disabled, skipping evaluation.")
