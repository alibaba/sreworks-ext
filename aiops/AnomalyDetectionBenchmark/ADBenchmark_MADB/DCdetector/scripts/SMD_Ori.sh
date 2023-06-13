export CUDA_VISIBLE_DEVICES=3

# index=

for i in {11..42};
do

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset SMD_Ori  --data_path SMD_Ori   --input_c 38 --output 38 --index $i --win_size 50  --patch_size 5
python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256    --mode test    --dataset SMD_Ori   --data_path SMD_Ori     --input_c 38   --output 38  --index $i  --win_size 50 --patch_size 5

done  