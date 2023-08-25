#!/bin/bash

cd experiments/ecg

threshold=0.02
fold_cnt=1

dataroot=$1
dataname=$2
model="beatgan"

w_adv=1
niter=100
lr=0.0001
n_aug=0

outf="./output"

for (( i=0; i<$fold_cnt; i+=1))
do
    echo "#################################"
    echo "########  Folder $i  ############"
    python -u test.py  \
        --dataroot $dataroot \
        --model $model \
        --niter $niter \
        --lr $lr \
        --outf  $outf \
        --folder $i \
        --dataname $dataname
done