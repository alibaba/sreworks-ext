export CUDA_VISIBLE_DEVICES=3

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 512  --mode train --dataset MSL  --data_path MSL_1d  --input_c 1 --output_c 1  --loss_fuc MSE 
python main.py --anormly_ratio 1  --num_epochs 10     --batch_size 512    --mode test    --dataset MSL   --data_path MSL_1d --input_c 1    --output_c 1  --loss_fuc MSE 
