import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--wandb-run-name', type=str, default=None)
parser.add_argument('--decision-method', type=str, default='topk_loose') # topk topk_semi topk_loose similarity
parser.add_argument('--sim-threshold', type=float, default=0.9)
parser.add_argument('--decision-k', type=int, default=10)
parser.add_argument('--decision-k-sub', type=int, default=5)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
    # added 
    "decision_method": args.decision_method,
    "sim_threshold": args.sim_threshold,
    "decision_k": args.decision_k,
    "decision_k_sub": args.decision_k_sub
}





import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
from ..model.cnets import Model
from ..model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process:
    import wandb
    from datetime import datetime

    # 构造包含年月日和参数的名称
    run_name = (
    f"{datetime.now().strftime('%Y%m%d_%H%M')}_"
    f"decision_method-{args.decision_method}_"
    f"sim_threshold-{args.sim_threshold}_"
    f"decision_k-{args.decision_k}_"
    f"decision_k_sub-{args.decision_k_sub}"
    )



    wandb.init(
        project="SpecAlign", 
        entity="reflectionie", 
        config=train_config,
        name=run_name
    )

baseconfig = AutoConfig.from_pretrained(args.basepath)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head

@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values,
                                                        use_cache=True)
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)
                last_hidden = out_hidden[:, -1:]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)

        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    save_dir = os.path.join(args.cpdir, run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


config = EConfig.from_pretrained(train_config["config_path"])




###################################################for debug###################################################
if args.debug:
    from model.ea_model import EaModel
    model = EaModel.from_pretrained(
            base_model_path=args.basepath,
            ea_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
            # total_token=63,
            # depth=5,
            top_k=10,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto"
        )
    head = model.base_model.lm_head
    model = model.ea_layer
###################################################for debug###################################################
else:
    model = Model(config, load_emb=True, path=args.basepath)
    
    
#################for train from ckpt #################
if args.ckpt_path is not None: 
    ea_model_path = args.ckpt_path
    load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
    if os.path.exists(load_model_path):
        ea_layer_state_dict = torch.load(load_model_path, map_location="cuda")
    else:
        load_model_path = os.path.join(ea_model_path, "model.safetensors")
        ea_layer_state_dict = safetensors.torch.load_file(load_model_path)


    # 打印当前模型的keys
    print("Current model keys:", model.state_dict().keys())

    # 打印预训练模型的keys
    print("Pretrained keys:", ea_layer_state_dict.keys())
    model.load_state_dict(ea_layer_state_dict, strict=True)
    print(f"load model from {load_model_path}")
######################################################



criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )
    

def default_similarity_fn(x, y):
    """
    默认相似度函数：利用 cosine similarity 计算 x 与 y 的相似度，
    返回每个样本的相似度，形状为 [bs]。
    """
    return F.cosine_similarity(x, y, dim=1)

# 举例：你也可以定义其他相似度函数，例如用点积计算相似度（未归一化的版本）
def dot_similarity_fn(x, y):
    """
    使用点积作为相似度计算函数。
    """
    return (x * y).sum(dim=1)


# =========================
# semi top‑k 条件判定函数
# =========================
def semi_topk_condition(topk_pred: torch.Tensor,
                        topk_gt: torch.Tensor,
                        k_sub: int) -> torch.Tensor:
    """
    对于 [bs, seq_len, k] 的 top-k 预测/目标，
    若二者在相同索引位置上匹配的次数 >= k_sub 则判定成功。

    :param topk_pred: [batch_size, seq_len, k]，模型预测出的 top-k token（有序）
    :param topk_gt:   [batch_size, seq_len, k]，目标的 top-k token（有序）
    :param k_sub:     int，若匹配位置数 >= k_sub 则判定成功
    :return: condition: [batch_size, seq_len] 的布尔张量
    """

    # 逐位置比较 => matches: [bs, seq_len, k]，True/False
    matches = (topk_pred == topk_gt)

    # 统计“命中”的位置数 => overlap_count: [bs, seq_len]
    overlap_count = matches.sum(dim=-1)

    # 判定是否 >= k_sub => [bs, seq_len]
    return (overlap_count >= k_sub)



# =========================
# 松散 top‑k 条件判定函数
# =========================

def loose_topk_condition(topk_pred: torch.Tensor,
                         topk_gt: torch.Tensor,
                         k_sub: int) -> torch.Tensor:
    """
    对于每个样本、每个 time step，如果 topk_pred 与 topk_gt 在前 k 个 token 里
    有足够多 (>= k_sub) 的相同 token，则返回 True，否则返回 False。

    参数:
    - topk_pred: shape [bs, seq_len, k]，表示预测的 top-k token
    - topk_gt:   shape [bs, seq_len, k]，表示目标的 top-k token
    - k_sub:     int，要求的最小交集大小

    返回:
    - condition: shape [bs, seq_len] 的布尔张量
    """

    # 1) 先对最内层维度进行比较
    #    topk_pred.unsqueeze(-1):  [bs, seq_len, k, 1]
    #    topk_gt.unsqueeze(-2):    [bs, seq_len, 1, k]
    #    => matching: [bs, seq_len, k, k]，matching[..., i, j] = (topk_pred[..., i] == topk_gt[..., j])
    matching = (topk_pred.unsqueeze(-1) == topk_gt.unsqueeze(-2))

    # 2) 统计 intersection_size：即有多少对 token 相同
    #    sum(dim=(-1, -2)) 表示对 k×k 这个维度做求和 => 得到 [bs, seq_len]
    intersection_size = matching.sum(dim=(-1, -2))

    # 3) 判断是否 >= k_sub
    condition = (intersection_size >= k_sub)

    return condition

    
# accelerator.load_state("checkpoints/state_5")
for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    
    # ----------------------------
    # 训练循环
    # ----------------------------
    similarity_fn = default_similarity_fn  # 你自定义的函数

    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        # ----------------------------
        # 1) 初始前向 (无梯度)，得到全序列的预测
        # ----------------------------
        with torch.no_grad():
            # predict_init: [batch_size, seq_len, hidden_dim]
            predict_init = model(
                data["hidden_states"],
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"]
            )

            # target_head: [batch_size, seq_len, vocab_size]
            target_head = head(data["target"])  
            # target_p: [batch_size, seq_len, vocab_size]
            target_p = torch.softmax(target_head, dim=2).detach()

            # ----------------------------
            # 2) 计算「是否替换」的条件 (一次性处理，而非逐 token 循环)
            # ----------------------------
            batch_size, seq_len, hidden_dim = data["hidden_states"].shape
            device = data["hidden_states"].device

            # 初始化，默认不替换
            condition = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

            if train_config['decision_method'] in ["topk", "topk_semi", "topk_loose"]:
                # 一次性计算 logits / softmax
                # logits_all: [bs, seq_len, vocab_size]
                logits_all = head(predict_init)  
                
                # 如果你只需要比较 top-k 的索引，可以不必先做 softmax
                # 直接对 logits 做 torch.topk() 即可。这里只是给出完整写法。
                probs_all = torch.softmax(logits_all, dim=-1)  

                # 预测的 top-k: [bs, seq_len, k]
                topk_pred = torch.topk(probs_all, train_config['decision_k'], dim=-1).indices
                # 目标的 top-k: [bs, seq_len, k]
                topk_gt   = torch.topk(target_p, train_config['decision_k'], dim=-1).indices

                if train_config['decision_method'] == "topk":
                    # condition: [bs, seq_len]
                    condition = (topk_pred == topk_gt).all(dim=-1)
                elif train_config['decision_method'] == "topk_semi":
                    condition = semi_topk_condition(topk_pred, topk_gt, train_config['decision_k_sub'])
                elif train_config['decision_method'] == "topk_loose":
                    condition = loose_topk_condition(topk_pred, topk_gt, train_config['decision_k_sub'])
                else:
                    raise ValueError(f"Unsupported topk variant: {train_config['decision_method']}")

            elif train_config['decision_method'] == "similarity":
                # pred_hidden: [bs, seq_len, hidden_dim]
                pred_hidden = predict_init
                # gt_hidden:   [bs, seq_len, hidden_dim]
                gt_hidden   = data["hidden_states"]

                # similarity_fn(...) 要返回 [bs, seq_len] 的相似度
                similarity = similarity_fn(pred_hidden, gt_hidden)
                condition = (similarity > train_config['sim_threshold'])

            elif train_config['decision_method'] == "eagle":
                # "eagle" 不做任何替换，直接 pass
                condition = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

            else:
                raise ValueError(f"Unsupported decision_method: {train_config['decision_method']}")

            # 只对有效 token 才考虑替换
            # valid_token: [bs, seq_len]
            valid_token = (data["attention_mask"].bool() & data["loss_mask"].bool())
            condition = condition & valid_token

            # ----------------------------
            # 3) 统一替换 + 统计替换率
            # ----------------------------
            modified_hidden_states = data["hidden_states"].clone()  # [bs, seq_len, hidden_dim]

            # 将 condition 扩展到形状 [bs, seq_len, hidden_dim]
            cond3d = condition.unsqueeze(-1).expand(-1, -1, hidden_dim)
            # 一次性替换
            modified_hidden_states = torch.where(
                cond3d,
                predict_init.detach(),  # 用模型的预测隐藏态来替换
                modified_hidden_states
            )

            # 计算每个样本替换了多少 token
            replacement_count = condition.sum(dim=1).float()  # [bs]
            # 计算每个样本中需要预测的有效 token 数
            valid_lengths = (data["attention_mask"] * data["loss_mask"]).sum(dim=1).float()  # [bs]
            epsilon = 1e-8
            replacement_rate = replacement_count / (valid_lengths + epsilon)
            avg_replacement_rate = replacement_rate.mean().item()

        # ----------------------------
        # 4) 第二次前向传播（有梯度），计算 loss 并优化
        # ----------------------------
        predict = model(
            modified_hidden_states,
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"]
        )

        loss_mask = data["loss_mask"][:, :, None]
        vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

        accelerator.backward(loss)
        accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
        optimizer.step()

        if is_warmup:
            scheduler.step()



        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {
                    "avg_replacement_rate": avg_replacement_rate,
                    "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                    "train/vloss": vloss.item(),
                    "train/ploss": ploss.item(),
                    "train/loss": loss.item(),
                    "train/acc": cc / ct,
                }
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
            
        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if (epoch + 1) % train_config["save_freq"] == 0:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            state_output_dir = os.path.join(save_dir, f"state_{epoch}")
            accelerator.save_state(output_dir=state_output_dir)
