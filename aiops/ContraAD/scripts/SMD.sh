export CUDA_VISIBLE_DEVICES=3

accelerate launch main.py --anormly_ratio 0.65 --num_epochs 3   --batch_size 128  --mode train --dataset SMD  --data_path SMD   --input_c 38   --output_c 38  --loss_fuc MSE  --win_size 135  --patch_size 57
accelerate launch main.py --anormly_ratio 0.65 --num_epochs 1   --batch_size 128  --mode test    --dataset SMD   --data_path SMD     --input_c 38      --output_c 38   --loss_fuc MSE   --win_size 135  --patch_size 57