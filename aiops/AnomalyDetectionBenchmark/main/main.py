import argparse
import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys


def print_output(proc):
    for line in iter(proc.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


parser = argparse.ArgumentParser()

# Add all the parameters you need there
parser.add_argument('--model', type=str, default='DCDetector',help='model name')
parser.add_argument('--dataset', type=str, default='MSL', help='dataset name')
parser.add_argument('--result_save_path', type=str, default='result.csv')
parser.add_argument('--instance', type=int, default=14) # HOLO instance number

# other parameters
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)

# only for Anomaly-transformer and DCdetector
parser.add_argument('--input_c', type=int, default=55)
parser.add_argument('--output_c', type=int, default=55) 
parser.add_argument('--anormly_ratio', type=float, default=1.0)

config = parser.parse_args()
args = vars(config)


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
    command = "python ../models/DAGMM/main.py"
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

proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print_output(proc)

if config.model == 'DCDetector' or config.model == 'AnomalyTransformer':
    proc2 = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print_output(proc2)


