export CUDA_VISIBLE_DEVICES=1

python main.py --anormly_ratio 1.9 --num_epochs 3   --batch_size 128  --mode train --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 155  --patch_size 13
python main.py --anormly_ratio 1.9  --num_epochs 10     --batch_size 64   --mode test    --dataset NIPS_TS_Swan   --data_path NIPS_TS_Swan --input_c 38    --output_c 38    --loss_fuc MSE       --win_size 155   --patch_size 13

accelerate launch main.py --anormly_ratio 1.9 --num_epochs 3   --batch_size 128  --mode train --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 155  --patch_size 13
accelerate launch main.py --anormly_ratio 1.9 --num_epochs 3   --batch_size 64  --mode test --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 155  --patch_size 13
