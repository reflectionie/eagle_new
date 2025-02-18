# multi_gpu_run.py
"""
PYTHONPATH=. python -m eagle.ge_data.diff_multi_gpu --ckpt_dir /net/papilio/storage7/tingyuan/llama/eagle_new/eagle/reflectio/train_data --ckpt_path_ea_model /net/graphium/storage3/tingyuan/models/EAGLE-LLaMA3-Instruct-8B/

PYTHONPATH=. python -m eagle.ge_data.diff_multi_gpu --ckpt_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/train_data   --ckpt_path_ea_model /home/5/uu02155/data/llama/HASS/refl/li/EAGLE-LLaMA3-Instruct-8B --num_gpus 4
"""
import os
import math
import argparse
from concurrent.futures import ThreadPoolExecutor

def run_command(cmd):
    print("[CMD]", cmd)
    os.system(cmd)

def split_range(total, n):
    """
    把 0 ~ total 切分成 n 段，返回 [(start, end), (start, end), ...]
    其中 end 是不包含的下标
    """
    chunk_size = math.ceil(total / n)
    result = []
    s = 0
    while s < total:
        e = min(s + chunk_size, total)
        result.append((s, e))
        s = e
    return result

parser = argparse.ArgumentParser(description="Launch multi-GPU ckpt process")
parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the ckpt directory")
parser.add_argument("--ckpt_path_ea_model", type=str, required=True, help="Path to the ea_model directory")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
args = parser.parse_args()

# 1) 列出所有 ckpt
all_ckpt_files = []
for root, dirs, files in os.walk(args.ckpt_dir):
    for f in files:
        if f.endswith(".ckpt"):
            all_ckpt_files.append(os.path.join(root, f))

all_ckpt_files = sorted(all_ckpt_files)
total = len(all_ckpt_files)
print(f"Total ckpt files = {total}")

# 2) 按 GPU 数量切分区间
intervals = split_range(total, args.num_gpus)

# 3) 给每个 GPU 分配一段
commands = []
for gpu_i, (start_idx, end_idx) in enumerate(intervals):
    cmd = f"PYTHONPATH=. python -m eagle.ge_data.idx_new_file_diff_ge_data_all_llama3 " \
          f"--ckpt_dir {args.ckpt_dir} " \
          f"--ckpt_path_ea_model {args.ckpt_path_ea_model} " \
          f"--gpu_index {gpu_i} " \
          f"--start_idx {start_idx} " \
          f"--end_idx {end_idx}"
    commands.append(cmd)

# 4) 并行执行
with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
    for cmd in commands:
        executor.submit(run_command, cmd)
