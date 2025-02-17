export CUDA_VISIBLE_DEVICES=0
accelerate launch main.py --anormly_ratio 0.6 --num_epochs 5  --batch_size 64  --mode train --dataset NIPS_TS_Water  --data_path NIPS_TS_Water  --input_c 9 --output_c 9  --loss_fuc MSE   --patch_size 135  --win_size 245
accelerate launch main.py --anormly_ratio 0.8  --num_epochs 3     --batch_size 64   --mode test    --dataset NIPS_TS_Water   --data_path NIPS_TS_Water --input_c 9    --output_c 9    --loss_fuc MSE   --patch_size 135   --win_size 245

python main.py --anormly_ratio 0.6  --num_epochs 3     --batch_size 16   --mode test    --dataset NIPS_TS_Water   --data_path NIPS_TS_Water --input_c 9    --output_c 9    --loss_fuc MSE   --patch_size 135   --win_size 135

