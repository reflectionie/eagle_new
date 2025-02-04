PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.main --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --bs 6
# 20 min * 20 epoch = 7 hours


PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method eagle --bs 6