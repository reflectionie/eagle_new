# 20 min * 20 epoch = 7 hours
# 14 hours
# 20 min * 20 epoch + 15min test * 4 = 

########02_04#########
# 0. baseline 3031222
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method eagle --bs 6

# 3031223
# 1. --decision-method为topk，--decision-k为10
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk --decision-k 10 --bs 6

# 3031224
# 2. --decision-method为topk，--decision-k为5
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk --decision-k 5 --bs 6

# 3031225
# 3. --decision-method为topk，--decision-k为1
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk --decision-k 1 --bs 6

# 3031227
# 4. --decision-method为topk_semi，--decision-k为10，--decision-k-sub为5
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk_semi --decision-k 10 --decision-k-sub 5 --bs 6

# 3031228
# 5. --decision-method为topk_semi，--decision-k为10，--decision-k-sub为3
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk_semi --decision-k 10 --decision-k-sub 3 --bs 6

# 3031229
# 6. --decision-method为topk_semi，--decision-k为10，--decision-k-sub为1
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk_semi --decision-k 10 --decision-k-sub 1 --bs 6

# 3031230
# 7. --decision-method为topk_loose，--decision-k为10，--decision-k-sub为5
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk_loose --decision-k 10 --decision-k-sub 5 --bs 6

# 3031233
# 8. --decision-method为topk_loose，--decision-k为10，--decision-k-sub为3
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk_loose --decision-k 10 --decision-k-sub 3 --bs 6

# 3031234
# 9. --decision-method为topk_loose，--decision-k为10，--decision-k-sub为1
PYTHONPATH=. ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 -m eagle.train.mainv2 --tmpdir ./eagle/reflectio/train_data/ --cpdir ./eagle/reflectio/checkpoints/ --basepath ./base_model/Meta-Llama-3-8B-Instruct/ --configpath ./eagle/train/EAGLE-LLaMA3-Instruct-8B.json --decision-method topk_loose --decision-k 10 --decision-k-sub 1 --bs 6

