export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.1 --num_epochs 3   --batch_size 128  --mode train --dataset NIPS_TS_CCard  --data_path NIPS_TS_CCard  --input_c 28 --output_c 28  --loss_fuc MSE   
python main.py --anormly_ratio 0.1  --num_epochs 10     --batch_size 128   --mode test    --dataset NIPS_TS_CCard   --data_path NIPS_TS_CCard --input_c 28    --output_c 28    --loss_fuc MSE  
