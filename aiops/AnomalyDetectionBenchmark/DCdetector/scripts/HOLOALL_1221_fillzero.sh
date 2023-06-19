#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


colnum_array=(15)

dataname_array=("instance0")

for index in in {1..2}
do 
    python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset HOLOALL  --data_path HOLO1221_fillzero --dataname ${dataname_array[$index]}  --input_c ${colnum_array[$index]}   --output_c ${colnum_array[$index]} --loss_fuc MSE
    python main.py --anormly_ratio 1  --num_epochs 10   --batch_size 256     --mode test    --dataset HOLOALL   --data_path HOLO1221_fillzero --dataname ${dataname_array[$index]} --input_c ${colnum_array[$index]}  --output_c ${colnum_array[$index]}  --loss_fuc MSE
done 




