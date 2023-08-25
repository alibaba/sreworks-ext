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


device = get_default_device()

param_grid = {
    'BATCH_SIZE': [320, 640, 1280],
    'N_EPOCHS': [50, 100, 150],
    'hidden_size': [100, 150, 200],
    'window_size': [12, 24, 48]
}

base_path ='../datasets/holo/'

data_paths = [base_path + 'filllinear_std', base_path + 'fillmean_std', base_path + 'fillzero_std']


for path in data_paths:
    datas = os.listdir(path)
    
    for data in datas:
        if data not in instances:
            continue
        normal_path = os.path.join(path, data, 'train.npy')
        attack_path = os.path.join(path, data, 'test.npy')
        label_path = os.path.join(path, data, 'test_label.npy')

        normal = np.load(normal_path)
        attack = np.load(attack_path)
        is_write_header = False
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
                scores = combine_all_evaluation_scores(labels, pred, loss, data + info)
                print(scores)
                with open(f'{path.split("/")[-1]}/{data}.csv','a',newline='') as f:
                    writer = csv.DictWriter(f,fieldnames=scores.keys())
                    if not is_write_header:
                        writer.writeheader()
                        is_write_header = True
                    writer.writerow(scores)
            except:
                pass
