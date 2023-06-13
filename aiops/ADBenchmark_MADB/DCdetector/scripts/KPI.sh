export CUDA_VISIBLE_DEVICES=1

python main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 32  --mode train --dataset KPI  --data_path KPI  --input_c 1 --output_c 1  --loss_fuc MSE   --win_size 60 --patch_size 5
python main.py --anormly_ratio 0.5  --num_epochs 10     --batch_size 32   --mode test    --dataset KPI   --data_path KPI --input_c 1    --output_c 1    --loss_fuc MSE    --win_size 60 --patch_size 5
