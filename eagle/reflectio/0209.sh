#!/bin/bash
# 修改脚本后，请确保此脚本有可执行权限：chmod +x your_script.sh

# 切换到项目根目录
cd ../..

##############################################
# 任务 2：decision-method 为 topk，decision-k 为 5
echo "Starting training with decision-method: topk (decision-k 5)"
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m eagle.train.mainv2 \
  --tmpdir ./eagle/reflectio/train_data/ \
  --cpdir ./eagle/reflectio/checkpoints/ \
  --basepath ./base_model/Meta-Llama-3-8B-Instruct/ \
  --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
  --decision-method topk \
  --decision-k 5 \
  --bs 6 \
  --ckpt_path /home/5/uu02155/data/llama/hass_new/reflectio/li/EAGLE-LLaMA3-Instruct-8B \
  --lr 0.00001

# 等待一定时间，确保 GPU 内存释放
echo "Waiting 300 seconds for GPU memory to be released..."
sleep 300

##############################################
# 任务 3：decision-method 为 topk，decision-k 为 10
echo "Starting training with decision-method: topk (decision-k 10)"
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m eagle.train.mainv2 \
  --tmpdir ./eagle/reflectio/train_data/ \
  --cpdir ./eagle/reflectio/checkpoints/ \
  --basepath ./base_model/Meta-Llama-3-8B-Instruct/ \
  --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
  --decision-method topk \
  --decision-k 10 \
  --bs 6 \
  --ckpt_path /home/5/uu02155/data/llama/hass_new/reflectio/li/EAGLE-LLaMA3-Instruct-8B \
  --lr 0.00001

echo "All training tasks finished."
