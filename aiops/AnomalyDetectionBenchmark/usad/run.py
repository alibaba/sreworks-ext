import torch.utils.data as data_utils
import numpy as np
import pdb
import torch
import torch.nn as nn
import csv


from utils import *
from usad import *
from spot import SPOT
from metrics.combine_all_scores import combine_all_evaluation_scores

def adjust_predicts(score, label,
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



device = get_default_device()

BATCH_SIZE =  320
N_EPOCHS = 100
hidden_size = 100
import os
datas =['instance name']
for data in datas:
    normal_path = "train.npy"
    attack_path = "test.npy"
    label_path = f"test_label.npy"

    normal = np.load(normal_path)
    attack = np.load(attack_path)

    window_size=12

    windows_normal=normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
    print(windows_normal.shape)
    windows_attack=attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
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
        scores = combine_all_evaluation_scores(labels, pred, loss, data)
        print(scores)
        with open('new_fillzero.csv','a',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=scores.keys())
            writer.writerow(scores)
    except:
        pass
