PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16  \
  -m eagle.train.main_mmd \
  --tmpdir ./eagle/reflectio/train_data/ \
  --cpdir ./eagle/reflectio/checkpoints/ \
  --basepath ./base_model/Meta-Llama-3-8B-Instruct/ \
  --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
  --gradient-accumulation-steps 2 \
  --bs 2 \
  --lr 0.00001 \
  --w1 1.0 \
  --w2 1.0 \
  --resume_checkpoint ./w1:_1.0_w2:_1.0_20250225_1531_./eagle/reflectio/checkpoints/state_2


PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16  \
  -m eagle.train.main_mmd \
  --tmpdir ./eagle/reflectio/train_data/ \
  --cpdir ./eagle/reflectio/checkpoints/ \
  --basepath ./base_model/Meta-Llama-3-8B-Instruct/ \
  --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
  --gradient-accumulation-steps 2 \
  --bs 2 \
  --lr 0.00001 \
  --w1 0 \
  --w2 0 \
  --resume_checkpoint ./w1:_0.0_w2:_0.0_20250225_1532_./eagle/reflectio/checkpoints/state_2

PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16  \
  -m eagle.train.main_mmd \
  --tmpdir ./eagle/reflectio/train_data/ \
  --cpdir ./eagle/reflectio/checkpoints/ \
  --basepath ./base_model/Meta-Llama-3-8B-Instruct/ \
  --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
  --gradient-accumulation-steps 2 \
  --bs 2 \
  --lr 0.00001 \
  --w1 1.0 \
  --w2 0 \
  --resume_checkpoint ./w1:_1.0_w2:_0.0_20250225_1532_./eagle/reflectio/checkpoints/state_2

PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16  \
  -m eagle.train.main_mmd \
  --tmpdir ./eagle/reflectio/train_data/ \
  --cpdir ./eagle/reflectio/checkpoints/ \
  --basepath ./base_model/Meta-Llama-3-8B-Instruct/ \
  --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json \
  --gradient-accumulation-steps 2 \
  --bs 2 \
  --lr 0.00001 \
  --w1 0 \
  --w2 1.0 \
  --resume_checkpoint ./w1:_0.0_w2:_1.0_20250225_1532_./eagle/reflectio/checkpoints/state_2

