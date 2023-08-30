import torch.utils.data as data_utils
import numpy as np
import pdb
import torch
import torch.nn as nn
import csv

import itertools
import os
from utils import *
from usad import *
from spot import SPOT
from metrics.combine_all_scores import combine_all_evaluation_scores

def adjust_predicts(score, 
                    label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def _generate_param_combinations(param_grid):
    keys, values = param_grid.keys(), param_grid.values()
    #keys, values = zip(*param_grid)
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_combinations

parser = argparse.ArgumentParser()

# Add all the parameters you need there

parser.add_argument('--dataset', metavar='-d', type=str, required=False, default='MSL', help='dataset name')
parser.add_argument('--instance', metavar='-i', type=str, required=False, default='15', help='instance number')
parser.add_argument('--holo_datafolder', type=str, default='../../datasets/holo/fillzero_std',help='holo_datafolder')
parser.add_argument('--public_datafolder', type=str, default='../../datasets/public/',help='public_datafolder')
parser.add_argument('--holo_result_save_path', type=str, default='../../result/holo_result.csv',help='holo_result_save_path')
parser.add_argument('--public_result_save_path', type=str, default='../../result/public_result.csv',help='public_result_save_path')
config = parser.parse_args()

public_datafolder = config.public_datafolder
public_datasets = config.dataset
holo_datafolder = config.holo_datafolder
holo_datasets = "instance"+config.instance
holo_result_file = config.holo_result_save_path
pub_result_file = config.public_result_save_path

device = get_default_device()

param_grid = {
    'BATCH_SIZE': [320, 640, 1280],
    'N_EPOCHS': [50, 100, 150],
    'hidden_size': [100, 150, 200],
    'window_size': [12, 24, 48]
}

#base_path ='../datasets/holo/'

#data_paths = [base_path + 'filllinear_std', base_path + 'fillmean_std', base_path + 'fillzero_std']

data_paths = [os.path.join(public_datafolder, public_datasets), os.path.join(holo_datafolder, holo_datasets)]


for i, path in enumerate(data_paths):
    
    normal_path = os.path.join(path, 'train.npy')
    attack_path = os.path.join(path, 'test.npy')
    label_path = os.path.join(path, 'test_label.npy')

    normal = np.load(normal_path)
    attack = np.load(attack_path)
    for params in _generate_param_combinations(param_grid):

        BATCH_SIZE = params['BATCH_SIZE']
        N_EPOCHS = params['N_EPOCHS']
        hidden_size = params['hidden_size']
        window_size = params['window_size']

        windows_normal=normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size + 1)[:, None]]
        print(windows_normal.shape)
        windows_attack=attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size + 1)[:, None]]
        print(windows_attack.shape)

        w_size=windows_normal.shape[1]*windows_normal.shape[2]
        z_size=windows_normal.shape[1]*hidden_size


        windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
        windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = UsadModel(w_size, z_size)

        model = to_device(model,device)

        history = training(N_EPOCHS,model,train_loader,val_loader)

        torch.save({
                    'encoder': model.encoder.state_dict(),
                    'decoder1': model.decoder1.state_dict(),
                    'decoder2': model.decoder2.state_dict()
                    }, "model.pth")

        checkpoint = torch.load("model.pth")

        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])

        results=testing(model,test_loader)
        loss=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                      results[-1].flatten().detach().cpu().numpy()])

        resultsT=testing(model,train_loader)
        lossT = np.concatenate([torch.stack(resultsT[:-1]).flatten().detach().cpu().numpy(),
                                      resultsT[-1].flatten().detach().cpu().numpy()])


        labels = np.load(label_path)[:len(loss)].flatten()
        print(loss.shape, lossT.shape)
        try:
            lms = 0.9999
            s = SPOT(1e-3)  # SPOT object
            s.fit(lossT, loss)  # data import
            s.initialize(level=0.999, min_extrema=False, verbose=False)  # initialization step

            ret = s.run(dynamic=False)  # run
            pot_th = np.mean(ret['thresholds']) * 1.0
            print(pot_th)
            pred, p_latency = adjust_predicts(loss, labels, pot_th, calc_latency=True)
            loss = np.array(loss)
            info = f"_{BATCH_SIZE}_{N_EPOCHS}_{hidden_size}_{window_size}"
            scores = combine_all_evaluation_scores(labels, pred, loss, "usad" + info)
            scores["instance"] = holo_datasets if i & 1 else public_datasets
            print(scores)
            tmp_path = holo_result_file if i & 1 else pub_result_file
            with open(tmp_path,'a',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=scores.keys())
                writer.writerow(scores)
        except:
            pass
