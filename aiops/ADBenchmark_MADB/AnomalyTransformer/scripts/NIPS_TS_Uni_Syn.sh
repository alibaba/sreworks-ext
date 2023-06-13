export CUDA_VISIBLE_DEVICES=0



python main.py --anormly_ratio 0.95 --num_epochs 3   --batch_size 32  --mode train --dataset NIPS-TS-Uni-Syn --data_path dataset/NIPS-TS-Uni-Syn  --input_c 1 --output_c 1 
python main.py --anormly_ratio 0.95  --num_epochs 3     --batch_size 32   --mode test  --dataset NIPS-TS-Uni-Syn --data_path dataset/NIPS-TS-Uni-Syn --input_c 1    --output_c 1    

