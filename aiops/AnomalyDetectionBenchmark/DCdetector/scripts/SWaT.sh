export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 3 --num_epochs 3   --batch_size 32  --mode train --dataset SWaT  --data_path SWaT  --input_c 1 --output_c 1  --loss_fuc MSE   --win_size 15 --patch_size 15
python main.py --anormly_ratio 3  --num_epochs 10     --batch_size 32   --mode test    --dataset SWaT   --data_path SWaT --input_c 1    --output_c 1    --loss_fuc MSE    --win_size 15 --patch_size 15