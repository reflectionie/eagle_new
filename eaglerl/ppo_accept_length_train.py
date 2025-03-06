#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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
        return {
            "input_ids": data["input_ids"],
            "loss_mask": data["loss_mask"]
        }

def get_allowed_prefix_lengths(loss_mask: torch.Tensor):
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
# EA 模型
# ------------------------
from eagle.model.ea_model_rl import EaModel

class AcceptLengthRewardModel(nn.Module):
    def __init__(self, policy_model: nn.Module, temperature: float, max_new_tokens: int, aggregator: str = "average"):
        super().__init__()
        self.policy_model = policy_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.aggregator = aggregator.lower()

    def forward(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        rewards = []
        seq_lengths = []
        for i in range(batch_size):
            ids = input_ids[i].unsqueeze(0)
            allowed_prefixes = get_allowed_prefix_lengths(loss_mask[i])
            sample_rewards = []
            sample_seq_lens = []
            for prefix_length in allowed_prefixes:
                prefix = ids[:, :prefix_length]
                self.policy_model.eval()
                with torch.no_grad():
                    _, _, _, accept_length = self.policy_model.eagenerate_rl(
                        prefix,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        log=False
                    )
                sample_rewards.append(accept_length)
                sample_seq_lens.append(prefix.shape[1] + self.max_new_tokens)
            if len(sample_rewards)==0:
                final_reward=0.0
                final_seq_len=ids.shape[1]
            else:
                if self.aggregator=="average":
                    final_reward = sum(sample_rewards)/len(sample_rewards)
                elif self.aggregator=="sum":
                    final_reward = sum(sample_rewards)
                elif self.aggregator=="max":
                    final_reward = max(sample_rewards)
                elif self.aggregator=="min":
                    final_reward = min(sample_rewards)
                else:
                    raise ValueError(f"Invalid aggregator: {self.aggregator}")
                final_seq_len = max(sample_seq_lens)
            rewards.append(final_reward)
            seq_lengths.append(final_seq_len)
        rewards = torch.tensor(rewards, device=input_ids.device, dtype=torch.float)
        seq_lengths = torch.tensor(seq_lengths, device=input_ids.device, dtype=torch.long)
        dummy_logits = torch.zeros(batch_size, seq_len, device=input_ids.device, dtype=torch.float)
        return dummy_logits, rewards, seq_lengths
    
    


# ------------------------
# 关键: EaModelWithValueHead, 兼容"two-model approach"的接口
# ------------------------
from transformers.modeling_outputs import BaseModelOutput
class EaModelWithValueHead(EaModel):
    """
    让其既能做policy，也能做value model:
     - base_model_prefix = "base_model"
     - forward(...) -> BaseModelOutput(hidden_states=...) for critic
     - score(...) -> linear value_head
    最后在 PPOTrainer 中, policy_model=value_model=同一个对象, 共享权重.
    """
    base_model_prefix = "base_model"  # PPOTrainer会 getattr(value_model, value_model.base_model_prefix)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 加一个价值头
        self.value_head = nn.Linear(self.hidden_size, 1)
        # 准备 generation_config
        if not hasattr(self, "generation_config"):
            self.generation_config = type("DummyGenConfig", (), {})()
            eos_id = self.get_tokenizer().eos_token_id
            if eos_id is None:
                eos_id = 2
            self.generation_config.eos_token_id = eos_id

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
        if not hasattr(new_obj, "generation_config"):
            new_obj.generation_config = type("DummyGenConfig", (), {})()
            eos_id = new_obj.get_tokenizer().eos_token_id or 2
            new_obj.generation_config.eos_token_id = eos_id
        return new_obj

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        注意: PPOTrainer的"two-model approach"对价值模型的调用逻辑是:
          output = critic_backbone(**kwargs)  # => should return BaseModelOutput with hidden_states
          value = value_model.score(output.hidden_states[-1])
        同时, PPOTrainer对policy只要 policy(...) -> logits 就行.

        => 我们这里做"伪two-model approach" with single object:
           - forward -> 返回 BaseModelOutput(hidden_states= [final_hidden_states])  (或  hidden_states=some_list )
           - score(...) -> 用 value_head
           - policy logits 仍可在别的地方(rollout)用 self.base_model.lm_head
        """
        # EaModel的super().forward => (outputs, hidden_states)
        outputs, hidden_states = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        # 返回一个 huggingface-style BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,  # (B, seq_len, hidden_dim)
            hidden_states=[hidden_states]      # PPOTrainer默认取 output.hidden_states[-1]
        )

    def score(self, last_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        PPOTrainer中:
          value = value_model.score(output.hidden_states[-1])
        => last_hidden_states 通常是 (B, seq_len, hidden_dim). 取最后token
        """
        if last_hidden_states.dim() == 3:
            # 取最后一个token
            last_hidden_states = last_hidden_states[:, -1, :]  # (B, hidden_dim)
        return self.value_head(last_hidden_states)  # (B,1)

    def get_policy_output(self, input_ids, attention_mask=None):
        """
        额外写一个方法给policy算logits. (rollout时, PPOTrainer自动?)
        这里可在集成  -> PPOTrainer大多只rollout(回答)用 `model.generate`
        """
        self.eval()
        with torch.no_grad():
            outs, hidden_states = super().forward(input_ids, attention_mask=attention_mask)
            logits = self.base_model.lm_head(hidden_states)
        return logits

# ------------------------
# 自定义 collate 函数，通过闭包获取 tokenizer
# ------------------------
def get_custom_collate_fn(tokenizer):
    def collate_fn(batch):
        # 将每个样本的 input_ids 和 loss_mask 转换为 tensor
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        loss_mask = [torch.tensor(item["loss_mask"], dtype=torch.long) for item in batch]
        # 使用 pad_sequence 进行 padding
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        padded_loss_mask = torch.nn.utils.rnn.pad_sequence(
            loss_mask, batch_first=True, padding_value=0
        )
        return {"input_ids": padded_input_ids, "loss_mask": padded_loss_mask}
    return collate_fn


# ------------------------
# 训练
# ------------------------
from transformers import HfArgumentParser
from trl import PPOTrainer, PPOConfig, ScriptArguments, ModelConfig, get_peft_config

def main():
    parser = argparse.ArgumentParser(description="伪two-model模式: policy+value 同一对象, 解决base_model_prefix问题")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32","float64","float16","bfloat16"])
    parser.add_argument("--aggregator", type=str, default="average",
                        help="聚合方式: average, max, min, sum")
    custom_args, remaining_argv = parser.parse_known_args()

    # parse PPO/training config
    hf_parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = hf_parser.parse_args_into_dataclasses(remaining_argv)
    dtype = str_to_torch_dtype(custom_args.dtype)

    # 1. 加载EaModelWithValueHead, 同时当policy & value
    model = EaModelWithValueHead.from_pretrained(
        base_model_path=custom_args.base_model_path,
        ea_model_path=custom_args.ea_model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        total_token=60, depth=5, top_k=10, threshold=1.0
    )
    tokenizer = model.get_tokenizer()
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = get_custom_collate_fn(tokenizer)

    model.to(custom_args.device)
    model.train()

    # 2. 数据
    dataset = CustomDataset(custom_args.datapath)

    # 3. 自定义reward
    reward_model = AcceptLengthRewardModel(
        policy_model=model,
        temperature=custom_args.temperature,
        max_new_tokens=custom_args.max_new_tokens,
        aggregator=custom_args.aggregator
    )
    reward_model.to(custom_args.device)

    # 4. 参考模型
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        # 参考模型，不做梯度更新
        ref_policy = EaModel.from_pretrained_w_base_model(
            base_model_path=custom_args.base_model_path,
            ea_model_path=custom_args.ea_model_path,
            total_token=60,
            depth=5,
            top_k=10,
            threshold=1.0,
            torch_dtype=dtype,
            device_map=model.base_model.device,
            low_cpu_mem_usage=True,
            base_model=model.base_model
        )
        ref_policy.to(custom_args.device)
    else:
        ref_policy = None

    # 5. PPOTrainer, 重点: model=..., value_model=..., 传同一个对象
    from trl import PPOTrainer
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,                # policy
        ref_model=ref_policy,       # KL
        reward_model=reward_model,
        value_model=model,          # 同一个对象 => 解决"NoneType"报错
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
        data_collator=data_collator
    )

    # 6. 训练
    trainer.train()

    # 7. 保存
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    trainer.generate_completions()

if __name__ == "__main__":
    main()
