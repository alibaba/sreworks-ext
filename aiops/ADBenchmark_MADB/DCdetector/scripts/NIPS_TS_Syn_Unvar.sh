export CUDA_VISIBLE_DEVICES=0



python main.py --anormly_ratio 3 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Syn_Unvar  --data_path NIPS_TS_Syn_Unvar  --input_c 1 --output_c 1  --loss_fuc MSE  
python main.py --anormly_ratio 3  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Syn_Unvar   --data_path NIPS_TS_Syn_Unvar --input_c 1    --output_c 1    --loss_fuc MSE    

