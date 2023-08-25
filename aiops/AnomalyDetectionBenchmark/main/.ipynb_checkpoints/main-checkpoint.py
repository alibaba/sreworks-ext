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
    command = "bash ./models/DCdetector/scripts/MSL.sh"
elif config.model == 'AnomalyTransformer':
    command = "bash ./models/AnomalyTransformer/scripts/MSL.sh"
elif config.model == 'ECOD':
    command = "python ./models/classic/main.py ECOD"
elif config.model == 'USAD':
    command = "python ./models/usad/main.py"
elif config.model == 'COPOD':
    command = "python ./models/classic/main.py COPOD"
elif config.model == 'BeatGAN':
    command = "python ./models/BeatGAN/main.py"
elif config.model == 'LSTM-VAE':
    command = "python ./models/classic/main.py LSTM-VAE"
elif config.model == 'DAGMM':
    command = "your_command_here"
elif config.model == 'DeepSVDD':
    command = "python ./models/classic/main.py DeepSVDD"
elif config.model == 'LSTM-AE':
    command = "python ./models/classic/main.py LSTM-AE"
elif config.model == 'LSTM':
    command = "python ./models/classic/main.py LSTM"
elif config.model == 'IForest':
    command = "python ./models/classic/main.py IForest"
elif config.model == 'KNN':
    command = "python ./models/classic/main.py KNN"
elif config.model == 'LOF':
    command = "python ./models/classic/main.py LOF"

proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print_output(proc)
