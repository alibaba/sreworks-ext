import argparse
import warnings
warnings.filterwarnings('ignore')
import subprocess
import os
import csv


def print_output(proc):
    for line in iter(proc.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

## 结果保存文件头写入
def write_head(file_path,pub):
    if os.path.exists(file_path):
        return
    if pub:
        head = "dataset"
    else:
        head = "instance"
    head_list = {
        "model": "model",
        head: "dataset",
        "Affiliation precision": "Affiliation precision",
        "Affiliation recall": "Affiliation recall",
        "MCC_score": "MCC_score",
        "R_AUC_PR": "R_AUC_PR",
        "R_AUC_ROC": "R_AUC_ROC",
        "VUS_PR": "VUS_PR",
        "VUS_ROC": "VUS_ROC",
        "f05_score_ori": "f05_score_ori",
        "f1_score_c": "f1_score_c",
        "f1_score_ori": "f1_score_ori",
        "f1_score_pa": "f1_score_pa",
        "pa_accuracy": "pa_accuracy",
        "pa_f_score": "pa_f_score",
        "pa_precision": "pa_precision",
        "pa_recall": "pa_recall",
        "point_auc": "point_auc",
        "precision_k": "precision_k",
        "range_auc": "range_auc",
        "range_f_score": "range_f_score",
    }
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=head_list.keys())
        writer.writeheader()
    return

holo_result_save_path = '../result/holo_result.csv'
public_result_save_path = '../result/public_result.csv'

public_datafolder = '../../datasets/public/'
holo_datafolder = '../../datasets/holo/fillzero_std'

parser = argparse.ArgumentParser()

# Add all the parameters you need there
parser.add_argument('--model', metavar='-m', type=str, required=False, default='DCDetector',help='model name')
parser.add_argument('--dataset', metavar='-d', type=str, required=False, default='MSL', help='dataset name')
parser.add_argument('--instance', metavar='-i', type=str, required=False, default='14', help='instance number')

# other parameters
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)

# only for Anomaly-transformer and DCdetector
parser.add_argument('--input_c', type=int, default=55)
parser.add_argument('--output_c', type=int, default=55) 
parser.add_argument('--anormly_ratio', type=float, default=1.0)

config = parser.parse_args()
args = vars(config)
config_command = f' --holo_result_save_path ../{holo_result_save_path} --public_result_save_path ../{public_result_save_path}'
config_command += f' --public_datafolder {public_datafolder} --holo_datafolder {holo_datafolder}'
config_command += f' --dataset {config.dataset} --instance {config.instance}'

if config.model == 'DCDetector':
    command =  "python main.py" + " --anormly_ratio " + str(config.anormly_ratio) + " --num_epochs " + str(config.num_epochs) + " --dataset " + config.dataset + " --batch_size " + str(config.batch_size)  + " --mode train " + " --data_path " + config.dataset + " --input_c " + str(config.input_c)  + " --output_c " + str(config.output_c) + " --result_save_path " + config.result_save_path + " --instance " + str(config.instance)
    command2 = "python main.py" + " --anormly_ratio " + str(config.anormly_ratio) + " --num_epochs " + str(config.num_epochs) + " --dataset " + config.dataset + " --batch_size " + str(config.batch_size)  + " --mode test " +  " --data_path " + config.dataset + " --input_c " + str(config.input_c)  + " --output_c " + str(config.output_c) + " --result_save_path " + config.result_save_path + " --instance " + str(config.instance)
elif config.model == 'AnomalyTransformer':
    command =  "python main.py" + " --anormly_ratio " + str(config.anormly_ratio) + " --num_epochs " + str(config.num_epochs) + " --dataset " + config.dataset + " --batch_size " + str(config.batch_size)  + " --mode train " + " --data_path " + config.dataset + " --input_c " + str(config.input_c)  + " --output_c " + str(config.output_c) + " --result_save_path " + config.result_save_path + " --instance " + str(config.instance)
    command2 = "python main.py" + " --anormly_ratio " + str(config.anormly_ratio) + " --num_epochs " + str(config.num_epochs) + " --dataset " + config.dataset + " --batch_size " + str(config.batch_size)  + " --mode test " +  " --data_path " + config.dataset + " --input_c " + str(config.input_c)  + " --output_c " + str(config.output_c) + " --result_save_path " + config.result_save_path + " --instance " + str(config.instance)
elif config.model == 'ECOD':
    command = "python ../models/classic/main.py ECOD"
elif config.model == 'USAD':
    command = "python ../models/usad/main.py"
elif config.model == 'COPOD':
    command = "python ../models/classic/main.py COPOD"
elif config.model == 'BeatGAN':
    command = "python ../models/BeatGAN/main.py"
elif config.model == 'LSTM-VAE':
    command = "python ../models/classic/main.py LSTM-VAE"
elif config.model == 'DAGMM':
    command = "python ../models/DAGMM/main.py" + config_command
elif config.model == 'DeepSVDD':
    command = "python ../models/classic/main.py DeepSVDD"
elif config.model == 'LSTM-AE':
    command = "python ../models/classic/main.py LSTM-AE"
elif config.model == 'LSTM':
    command = "python ../models/classic/main.py LSTM"
elif config.model == 'IForest':
    command = "python ../models/classic/main.py IForest"
elif config.model == 'KNN':
    command = "python ../models/classic/main.py KNN"
elif config.model == 'LOF':
    command = "python ../models/classic/main.py LOF"

write_head(holo_result_save_path, False)
write_head(public_result_save_path, True)

proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print_output(proc)

if config.model == 'DCDetector' or config.model == 'AnomalyTransformer':
    proc2 = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print_output(proc2)


