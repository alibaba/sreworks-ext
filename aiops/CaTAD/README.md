# Unveiling Time Series Anomaly Detection: A Causal Look and Solutions

This repo provides the implementation code corresponding to our paper entitled Unveiling Time Series Anomaly Detection: A Causal Look and Solutions. The code is implemented on Pytorch 1.10.2 on a server with NVIDIA A100.

![image](./img/framework.png)

## Description
We introduce a causal perspective to uncover the underlying mechanisms driving anomalies in time series data. Building on this foundation, we leverage causal tools to develop a new framework, CaTAD, aimed at enhancing both the accuracy and interpretability of Time Series Anomaly Detection (TSAD). To address the inherent challenges in TSAD, we design specialized modules that capture and fuse latent causal factors across different period levels, effectively managing local periodic information. Additionally, we propose a novel mask mechanism integrated into Normalizing Flows to account for global periodicity, enabling unsupervised, density-based anomaly detection.

## Requirements

CaTAD uses the following dependencies:

- Pytorch 1.8.1 and its dependencies
- Numpy and Scipy
- CUDA 11.1 or latest version, cuDNN

## Datasets
We evaluate the proposed model on seven real-world datasets from different domains:

- **Public Datasets**: MSL, SMD, PSM, WADI
- **Cloud Services Datasets**: Cloud-B, ODPS-SH, ODPS-Y

### Preparation
Ensure dataset folders are placed under `./data/`.

### SMD dataset
Download the dataset from:  https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

Place the files under `./data/SMD/`.

### PSM dataset
Download the dataset from:  https://github.com/eBay/RANSynCoders/tree/main/data

Place `train.csv`, `test.csv`, and `test_label.csv` in the `./data/PSM/` folder.


### MSL dataset
Download the dataset using the following commands:
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip
rm data.zip
cd data
wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

Ensure the folders ```train```, ```test```, and the file ```labeled_anomalies.csv``` are in the ```./data/MSL/``` folder.

### WADI dataset
You can get the dataset by filling out the request form at:
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Place the file ```WADI_attackdataLABLE.csv``` in the ```./data/``` floder.

## Training and Evaluation

Here is an example command for training on the SMD dataset:

```
python ./main.py --dataset SMD --mode 'train' --hid_dim 32 --n_blocks 5 --n_layers 2
```

## Code Reference

TS Forecasting via CNF: https://github.com/zalandoresearch/pytorch-ts

GANF: https://github.com/EnyanDai/GANF

Time-Series-Library: https://github.com/thuml/Time-Series-Library/

## Citation

If you find our work useful in your research, please cite:

```
Pending.
```
