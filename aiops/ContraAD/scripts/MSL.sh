export CUDA_VISIBLE_DEVICES=1

python main.py --anormly_ratio 0.75 --num_epochs 10   --batch_size 64  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 265  --patch_size 35
python main.py --anormly_ratio 0.75  --num_epochs 10     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 265  --patch_size 35


accelerate launch main.py --anormly_ratio 0.85 --num_epochs 5   --batch_size 32  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 265  --patch_size 35
#{'anomaly_ratio': 0.85, 'win_size': 55, 'accuracy': 0.9827544097693351, 'precision': 0.9291661160301309, 'recall': 0.9053566829770796, 'f_score': 0.9171068936281225, 'thre': 0.24239280819892883}
accelerate launch main.py --anormly_ratio 0.80  --num_epochs 1     --batch_size 32    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 265  --patch_size 35

accelerate launch main.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 8  --mode test --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 55  --patch_size 35



accelerate launch main.py --anormly_ratio 1 --num_epochs 1     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 60  --patch_size 35


accelerate launch  main.py  --num_epochs 5   --batch_size 256  --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 30   --mode test  --anormly_ratio 0.95
