#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from fastchat.utils import str_to_torch_dtype

# ------------------------
# 定义辅助函数与数据集
# ------------------------
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
        # 假设数据中包含 "input_ids" 和 "loss_mask"
        return {
            "input_ids": data["input_ids"],
            "loss_mask": data["loss_mask"]
        }

def get_allowed_prefix_lengths(loss_mask: torch.Tensor):
    """
    根据 1D loss_mask 返回允许的前缀长度列表：
      - 从开始连续为 0 的部分只允许取整个区间（request 部分）；
      - 对于后续 loss_mask==1 部分，每增加一个 token 都视为一个合法的前缀。
    """
    mask_list = loss_mask.tolist()
    allowed = []
    fixed = 0
    for token in mask_list:
        if token == 0:
            fixed += 1
        else:
            break
    allowed.append(fixed)
    total = len(mask_list)
    for j in range(fixed + 1, total + 1):
        allowed.append(j)
    return allowed

# ------------------------
# 导入 EA 模型（策略模型）
# ------------------------
from eagle.model.ea_model_rl import EaModel

# ------------------------
# 定义更新后的奖励模型
# ------------------------
import torch.nn as nn
class AcceptLengthRewardModel_update(nn.Module):
    def __init__(self, policy_model: nn.Module, temperature: float, max_new_tokens: int):
        """
        :param policy_model: EA 模型（策略模型），其 eagenerate_rl 返回 (generated, new_token, step, accept_length)
        :param temperature: 生成时使用的温度
        :param max_new_tokens: 生成的最大新 token 数
        """
        super().__init__()
        self.policy_model = policy_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def forward(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        """
        对于每个样本：
          1. 调用 get_allowed_prefix_lengths(loss_mask) 得到合法前缀长度列表；
          2. 对每个合法 prefix 调用 eagenerate_rl 得到 accept_length；
          3. 聚合这些 accept_length（此处取最大值）作为最终奖励。
        返回格式：(dummy_logits, rewards, sequence_lengths) 以符合 PPOTrainer 的接口要求。
        """
        batch_size, seq_len = input_ids.shape
        rewards = []
        seq_lengths = []
        for i in range(batch_size):
            ids = input_ids[i].unsqueeze(0)  # shape: [1, seq_len]
            allowed_prefixes = get_allowed_prefix_lengths(loss_mask[i])
            sample_rewards = []
            sample_seq_lens = []
            for prefix_length in allowed_prefixes:
                prefix = ids[:, :prefix_length]
                # 注意：为了不计算梯度，这里设为 eval 模式
                self.policy_model.eval()
                with torch.no_grad():
                    # 调用 eagenerate_rl，返回 (generated, new_token, step, accept_length)
                    _, _, _, accept_length = self.policy_model.eagenerate_rl(
                        prefix, 
                        temperature=self.temperature, 
                        max_new_tokens=self.max_new_tokens, 
                        log=False
                    )
                sample_rewards.append(accept_length)
                sample_seq_lens.append(prefix.shape[1] + self.max_new_tokens)
            final_reward = max(sample_rewards) if sample_rewards else 0.0
            rewards.append(final_reward)
            seq_lengths.append(max(sample_seq_lens) if sample_seq_lens else ids.shape[1])
        rewards = torch.tensor(rewards, device=input_ids.device, dtype=torch.float)
        seq_lengths = torch.tensor(seq_lengths, device=input_ids.device, dtype=torch.long)
        dummy_logits = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.float)
        return dummy_logits, rewards, seq_lengths

# ------------------------
# PPO训练部分：使用 trl 的 PPOTrainer
# ------------------------
from transformers import HfArgumentParser
from trl import PPOTrainer, PPOConfig, ScriptArguments, ModelConfig, get_peft_config, get_quantization_config, get_kbit_device_map, AutoModelForCausalLMWithValueHead

def main():
    # 解析我们自定义的基础参数
    parser = argparse.ArgumentParser(description="使用 PPO 训练 EA 模型最大化 accept_length")
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="基础模型所在目录（例如 Meta-Llama-3-8B-Instruct 的路径）")
    parser.add_argument("--ea-model-path", type=str, required=True,
                        help="EA 模型权重所在目录")
    parser.add_argument("--datapath", type=str, required=True,
                        help="训练数据集目录（包含 torch.save 文件）")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大新 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0 表示贪婪）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备，例如 'cuda' 或 'cpu'")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float64", "float16", "bfloat16"],
                        help="模型数据类型")
    # 注意: parse_known_args 可以返回 (已识别的参数, 未识别的剩余参数)
    custom_args, remaining_argv = parser.parse_known_args()

    # 使用 HfArgumentParser 解析 PPO/训练/模型相关参数
    hf_parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = hf_parser.parse_args_into_dataclasses(remaining_argv)

    # 加载 EA 模型（策略模型）
    torch_dtype = str_to_torch_dtype(custom_args.dtype)
    model = EaModel.from_pretrained(
        base_model_path=custom_args.base_model_path,
        ea_model_path=custom_args.ea_model_path,
        total_token=60,
        depth=5,
        top_k=10,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    model.to(custom_args.device)
    model.train()

    # 构造数据集与 DataLoader
    dataset = CustomDataset(custom_args.datapath)
    # 这里我们直接将 dataset 传递给 PPOTrainer，PPOTrainer 内部会创建 DataLoader
    # 如果需要可以在 Trainer 中进行进一步数据处理

    # 加载值函数模型：利用 reward_model_path 加载一个序列分类模型
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        custom_args.ea_model_path, trust_remote_code=model_args.trust_remote_code
    )
    value_model.to(custom_args.device)

    # 构造自定义奖励模型：使用更新后的 AcceptLengthRewardModel_update
    reward_model = AcceptLengthRewardModel_update(model, temperature=custom_args.temperature, max_new_tokens=custom_args.max_new_tokens)
    reward_model.to(custom_args.device)

    # 如果不使用 PEFT，则加载参考模型
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        from transformers import AutoModelForCausalLM
        ref_policy = EaModel.from_pretrained(
            base_model_path=custom_args.base_model_path,
            ea_model_path=custom_args.ea_model_path,
            total_token=60,
            depth=5,
            top_k=10,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        ref_policy.to(custom_args.device)
    else:
        ref_policy = None

    # 构造 PPOTrainer（注意：train_dataset 传入 dataset，eval_dataset 这里可设为 None 或另行构造）
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
    )

    # 开始 PPO 训练
    trainer.train()

    # 保存模型
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    # 生成样本预览
    trainer.generate_completions()

if __name__ == "__main__":
    main()
