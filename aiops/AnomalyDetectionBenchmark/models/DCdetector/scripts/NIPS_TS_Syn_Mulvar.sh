export CUDA_VISIBLE_DEVICES=0



python main.py --anormly_ratio 3 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Syn_Mulvar  --data_path NIPS_TS_Syn_Mulvar  --input_c 5 --output_c 5  --loss_fuc MSE  
python main.py --anormly_ratio 3  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Syn_Mulvar   --data_path NIPS_TS_Syn_Mulvar --input_c 5    --output_c 5    --loss_fuc MSE    

