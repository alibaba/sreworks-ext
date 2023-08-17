#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

colnum_array=(21
)

dataname_array=("instance0"
)

for index in in {1..2}
do 
    python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset HOLOALL  --data_path dataset/HOLO1221_fillzero --dataname ${dataname_array[$index]}  --input_c ${colnum_array[$index]}   --output_c ${colnum_array[$index]}
    python main.py --anormly_ratio 1  --num_epochs 10   --batch_size 256     --mode test    --dataset HOLOALL   --data_path dataset/HOLO1221_fillzero --dataname ${dataname_array[$index]} --input_c ${colnum_array[$index]}  --output_c ${colnum_array[$index]}  --pretrained_model 20
done 




