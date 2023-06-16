export CUDA_VISIBLE_DEVICES=2

python main.py --anormly_ratio 1  --num_epochs 3 --batch_size 256 --mode train --dataset MSL --data_path MSL_9d --input_c 9 --output_c 9 --loss_fuc MSE

python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset MSL --data_path MSL_9d --input_c 9 --output_c 9 --loss_fuc MSE

