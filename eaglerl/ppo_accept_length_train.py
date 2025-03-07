#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastchat.utils import str_to_torch_dtype

# ------------------------
# 1) 辅助函数
# ------------------------
def list_files(path):
    files = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            files.append(os.path.join(root, f))
    return files

def get_allowed_prefix_lengths(loss_mask: torch.Tensor):
    """
    返回所有合法前缀的长度列表
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
# 2) 自定义 Dataset: 每个前缀 => 一条样本
# ------------------------
class PrefixDataset(Dataset):
    """
    原始数据中 data["input_ids"], data["loss_mask"] 通常是一条完整序列。
    在这里把“所有合法前缀”都展开为多条记录，保存在 self.samples 里。
    这样 PPO 训练时，每个前缀就是一条单独的数据。
    """
    def __init__(self, datapath):
        super().__init__()
        self.samples = []
        files = list_files(datapath)
        for file_path in files:
            data = torch.load(file_path)
            # data["input_ids"], data["loss_mask"] 都是 list[int] 或 1D Tensor
            input_ids = data["input_ids"]
            loss_mask = data["loss_mask"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if not isinstance(loss_mask, torch.Tensor):
                loss_mask = torch.tensor(loss_mask, dtype=torch.long)
            # 枚举所有 prefix
            prefix_lengths = get_allowed_prefix_lengths(loss_mask)
            for prefix_len in prefix_lengths:
                # slice prefix
                prefix_ids = input_ids[:prefix_len]
                # 这里的 prefix_ids/ prefix_len 代表“一个独立前缀”
                # 当 PPOTrainer 拿到这条数据时，就会对 prefix_ids 做 roll-out
                # 也可以保留 prefix 的 loss_mask (取 [:prefix_len])，但不一定需要
                self.samples.append({
                    "input_ids": prefix_ids.tolist(),  # list[int], 便于后续 pad
                    # 不需要 aggregator，这里只存 prefix
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------
# 3) 自定义 RewardModel: 每条前缀只算一次accept_length
# ------------------------
class SinglePrefixRewardModel(nn.Module):
    """
    对于“已经是单个前缀”的数据，每次 forward 就调用 policy_model.eagenerate_rl(prefix)，
    直接得到 accept_length，作为这条数据的奖励 R。
    不需要聚合，因为每个prefix都被当作独立的样本。
    """
    def __init__(self, policy_model: nn.Module, temperature: float, max_new_tokens: int):
        super().__init__()
        self.policy_model = policy_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def forward(self, input_ids: torch.Tensor):
        """
        PPOTrainer 会传进 (input_ids, ), 里面 input_ids.shape=[batch_size, seq_len]
        这里的 seq_len 就是前缀长度 => 只做一次 eagenerate_rl。
        """
        batch_size, seq_len = input_ids.shape
        rewards = []
        seq_lengths = []

        # 对 batch 中每个 prefix 单独计算
        for i in range(batch_size):
            prefix = input_ids[i].unsqueeze(0)
            self.policy_model.eval()
            with torch.no_grad():
                # policy_model.eagenerate_rl(prefix) => (generated, new_token, step, accept_length)
                _, _, _, accept_length = self.policy_model.eagenerate_rl(
                    prefix,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    log=False
                )
            # 这里 prefix 是独立样本 => accept_length即这条数据的 reward
            rewards.append(float(accept_length))

            # 如果 PPOTrainer 需要 seq_lengths 来对齐，也可以指定
            seq_lengths.append(prefix.shape[1] + self.max_new_tokens)

        # 转回 tensor
        rewards = torch.tensor(rewards, dtype=torch.float, device=input_ids.device)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=input_ids.device)
        # PPOTrainer 要求 forward返回 (dummy_logits, rewards, seq_lengths)
        # dummy_logits 形状与 input_ids对应即可
        dummy_logits = torch.zeros_like(input_ids, dtype=torch.float)
        return dummy_logits, rewards, seq_lengths


# ------------------------
# 4) 定义EaModelWithValueHead => PPO “伪双模型模式” (共享一个对象做 policy+value)
# ------------------------
from transformers.modeling_outputs import BaseModelOutput
from eagle.model.ea_model_rl import EaModel

class EaModelWithValueHead(EaModel):
    base_model_prefix = "base_model"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_head = nn.Linear(self.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, base_model_path, ea_model_path, total_token=60, depth=5, top_k=10,
                        threshold=1.0, torch_dtype=None, device_map="auto", low_cpu_mem_usage=True, **kwargs):
        base_ea_model = super().from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs
        )
        new_obj = cls.__new__(cls)
        new_obj.__dict__ = base_ea_model.__dict__
        new_obj.value_head = nn.Linear(new_obj.hidden_size, 1)
        return new_obj

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # EaModel父类 => return (outputs, hidden_states)
        outputs, hidden_states = super().forward(input_ids, attention_mask, **kwargs)
        # 返回HF风格BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=[hidden_states],
        )

    def score(self, last_hidden_states: torch.Tensor) -> torch.Tensor:
        if last_hidden_states.dim() == 3:
            last_hidden_states = last_hidden_states[:, -1, :]
        return self.value_head(last_hidden_states)


# ------------------------
# 5) collate: pad prefix
# ------------------------
def get_custom_collate_fn(tokenizer):
    def collate_fn(batch):
        # batch里每个元素是 { "input_ids": list[int] }
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return {"input_ids": padded}
    return collate_fn


# ------------------------
# 6) 训练脚本
# ------------------------
from transformers import HfArgumentParser
from trl import PPOTrainer, PPOConfig, ScriptArguments, ModelConfig, get_peft_config

def main():
    parser = argparse.ArgumentParser(description="每个prefix独立当做样本的PPO训练示例")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32","float64","float16","bfloat16"])
    custom_args, remaining_argv = parser.parse_known_args()

    # parse PPO/training config
    hf_parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = hf_parser.parse_args_into_dataclasses(remaining_argv)
    from fastchat.utils import str_to_torch_dtype
    torch_dtype = str_to_torch_dtype(custom_args.dtype)

    # 1) 加载“伪双模型”EaModelWithValueHead
    model = EaModelWithValueHead.from_pretrained(
        base_model_path=custom_args.base_model_path,
        ea_model_path=custom_args.ea_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        total_token=60, depth=5, top_k=10, threshold=1.0
    )
    tokenizer = model.get_tokenizer()
    # 如无 pad_token，则添加
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model.to(custom_args.device)
    model.train()

    # 2) 数据: prefix级别
    dataset = PrefixDataset(custom_args.datapath)
    data_collator = get_custom_collate_fn(tokenizer)

    # 3) 构造RewardModel: 单prefix => accept_length
    reward_model = SinglePrefixRewardModel(
        policy_model=model,
        temperature=custom_args.temperature,
        max_new_tokens=custom_args.max_new_tokens,
    )
    reward_model.to(custom_args.device)

    # 4) 如果不用PEFT，则加载 ref_policy
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = EaModel.from_pretrained_w_base_model(
            base_model_path=custom_args.base_model_path,
            ea_model_path=custom_args.ea_model_path,
            total_token=60,
            depth=5,
            top_k=10,
            threshold=1.0,
            torch_dtype=torch_dtype,
            device_map=model.base_model.device,
            low_cpu_mem_usage=True,
            base_model=model.base_model
        )
        ref_policy.to(custom_args.device)
    else:
        ref_policy = None

    # 5) PPOTrainer
    #   policy与value都用同一个 model，避免多份权重
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=model,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    # 6) 训练
    trainer.train()

    # 7) 保存并生成预览
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    trainer.generate_completions()

if __name__ == "__main__":
    main()
