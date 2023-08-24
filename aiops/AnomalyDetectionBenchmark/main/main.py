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
# parser.add_argument('--dataset', type=str, default='MSL', help='dataset name')

# Other examples (different types)
# parser.add_argument('--win_size', type=int, default=100)
# parser.add_argument('--patch_size', type=list, default=[5])
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--rec_timeseries', action='store_true', default=True)
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

config = parser.parse_args()
args = vars(config)


if config.model == 'DCDetector':
    command = "bash ./model/DCdetector/scripts/MSL.sh"
elif config.model == 'AnomalyTransformer':
    command = "bash ./model/AnomalyTransformer/scripts/MSL.sh"
elif config.model == 'ECOD':
    command = "your_command_here"
elif config.model == 'USAD':
    command = "your_command_here"
elif config.model == 'COPOD':
    command = "your_command_here"
elif config.model == 'BeatGAN':
    command = "your_command_here"
elif config.model == 'LSTM-VAE':
    command = "your_command_here"
elif config.model == 'DAGMM':
    command = "your_command_here"
elif config.model == 'DeepSVDD':
    command = "your_command_here"
elif config.model == 'LSTM-AE':
    command = "your_command_here"
elif config.model == 'LSTM':
    command = "your_command_here"
elif config.model == 'IForest':
    command = "your_command_here"
elif config.model == 'KNN':
    command = "your_command_here"
elif config.model == 'LOF':
    command = "your_command_here"

proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print_output(proc)
