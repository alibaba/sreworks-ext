export CUDA_VISIBLE_DEVICES=1

python main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 128  --mode train --dataset Yahoo  --data_path Yahoo  --input_c 1 --output_c 1  --loss_fuc MSE   --win_size 60 --patch_size 5
python main.py --anormly_ratio 0.5  --num_epochs 10     --batch_size 128   --mode test    --dataset Yahoo   --data_path Yahoo --input_c 1    --output_c 1    --loss_fuc MSE    --win_size 60 --patch_size 5
