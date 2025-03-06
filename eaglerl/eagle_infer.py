#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader

# 导入模型相关代码
from eagle.model.ea_model import EaModel
# 如果需要可以使用 fastchat 中的工具转换 dtype（也可以直接用 torch.float16）
from fastchat.utils import str_to_torch_dtype

# 简单的文件遍历函数，用于获取数据文件列表
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
        # 返回字典形式，方便后续调用
        return {"input_ids": data["input_ids"]}

# 定义使用 eagenerate 的推理函数（内部使用 kv_cache）
def ea_forward(inputs, model, max_new_tokens, temperature=0.0, device="cuda"):
    input_ids = inputs["input_ids"]
    assert input_ids.shape[0] == 1, "目前只支持 batch size = 1"
    input_ids = input_ids.to(device)
    # 使用 eagenerate 进行生成（内部会重置 kv_cache 等）
    generated, new_token, step = model.eagenerate(
        input_ids, 
        temperature=temperature, 
        max_new_tokens=max_new_tokens, 
        log=True
    )

    return generated, new_token, step




def main():
    parser = argparse.ArgumentParser(description="使用训练数据集进行推理，并打印结果")
    parser.add_argument("--base-model-path", type=str, default="/net/papilio/storage7/tingyuan/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct",
                        help="基础模型所在目录")
    parser.add_argument("--ea_model_path", type=str, default="/net/graphium/storage3/tingyuan/models/EAGLE-LLaMA3-Instruct-8B",
                        help="EAGLE权重所在目录或仓库")
    parser.add_argument("--datapath", type=str, default="/net/papilio/storage7/tingyuan/llama/eagle_new/eagle/reflectio/train_data/eagle_data/sharegpt_0_300_mufp16",
                        help="训练数据集目录（例如存有 torch.save 文件的目录）")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大新 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度，0表示贪婪")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备，例如 'cuda' 或 'cpu'")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float64", "float16", "bfloat16"],
                        help="模型数据类型")
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

    # 构造数据集与 DataLoader（这里每个样本 batch size 为 1）
    dataset = CustomDataset(args.datapath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 对数据集中的第一个样本进行推理
    for sample in dataloader:
        # 解码原始 prompt
        prompt_text = tokenizer.decode(sample["input_ids"][0].tolist(), skip_special_tokens=True)
        print("Prompt:", prompt_text)
        generated, new_token, step = ea_forward(sample, model, args.max_new_tokens, args.temperature, args.device)
        output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print("Generated:", output_text)
        # 这里只处理第一个样本
        break

if __name__ == "__main__":
    main()
