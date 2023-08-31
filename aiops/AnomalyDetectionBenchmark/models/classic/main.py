import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from myutils import Utils
from torch.utils.data import DataLoader, TensorDataset

import itertools
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD
from network import LSTM, LSTM_AE, LSTM_VAE
from fit import fit_LSTM, fit_LSTM_AE, fit_LSTM_VAE
from eval import eval_LSTM, eval_LSTM_AE, eval_LSTM_VAE
from spot import SPOT
from metrics.combine_all_scores import combine_all_evaluation_scores
import csv


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


def write_ans(score_train, score_test, y_test, is_pub, dataset, name, save_path):
    try:
        lms = 0.9999
        s = SPOT(1e-3)  # SPOT object
        s.fit(score_train, score_test)  # data import
        s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step

        ret = s.run(dynamic=False)  # run
        pot_th = np.mean(ret['thresholds']) * 1.0
        print(pot_th)
        pred, p_latency = adjust_predicts(score_test, y_test, pot_th, calc_latency=True)

        y_test = np.array(y_test)
        score_test = np.array(score_test)
        scores = combine_all_evaluation_scores(y_test, pred, score_test, name)
        if is_pub:
            scores["instance"] = holo_datasets 
        else:
            scores["dataset"] = public_datasets
        print(scores)
        if os.path.exists(save_path):
            return
        if is_pub:
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
        with open(save_path,'a',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=scores.keys())
            writer.writerow(scores)
    except Exception as e:
        print(e)

def _generate_param_combinations(param_grid):
    keys, values = param_grid.keys(), param_grid.values()
    #keys, values = zip(*param_grid)
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_combinations

        

utils = Utils()


parser.add_argument('--model', required=True, default='KNN', help='model name')
parser.add_argument('--dataset', metavar='-d', type=str, required=False, default='MSL', help='dataset name')
parser.add_argument('--instance', metavar='-i', type=str, required=False, default='15', help='instance number')
parser.add_argument('--holo_datafolder', type=str, default='../../datasets/holo/fillzero_std',help='holo_datafolder')
parser.add_argument('--public_datafolder', type=str, default='../../datasets/public/',help='public_datafolder')
parser.add_argument('--holo_result_save_path', type=str, default='../../result/holo_result.csv',help='holo_result_save_path')
parser.add_argument('--public_result_save_path', type=str, default='../../result/public_result.csv',help='public_result_save_path')
config = parser.parse_args()

model_name = config.model
public_datafolder = config.public_datafolder
public_datasets = [config.dataset]
holo_datafolder = config.holo_datafolder
holo_datasets = "instance"+config.instance
holo_result_file = config.holo_result_save_path
pub_result_file = config.public_result_save_path

#data_paths = [os.path.join(public_datafolder, public_datasets), os.path.join(holo_datafolder, holo_datasets)]

if config.dataset != 'holo':
    is_pub = True
    save_path = pub_result_file
    datafolder = public_datafolder
    dataset = public_datasets
    path = os.path.join(public_datafolder, public_datasets)
else:
    is_pub = False
    save_path = holo_result_file
    datafolder = holo_datafolder
    dataset = holo_datasets
    path = os.path.join(holo_datafolder, holo_datasets)

utils.set_seed(42)
utils.get_device()


model_dict = {
            'KNN': KNN, 'LOF': LOF,
              'IForest': IForest,
              'COPOD': COPOD, 'ECOD': ECOD,
              'DeepSVDD': DeepSVDD,
              'LSTM': LSTM, 'LSTM_AE': LSTM_AE, 'LSTM_VAE': LSTM_VAE} # 'OCSVM': OCSVM


param_grid1 = {
    'n_estimators' : [100, 150, 200],
    'max_samples' : [128, 256, 512],
    'contamination' : [0.1, 0.15, 0.2]
}

param_grid2 = {
    'batch_size': [256, 512, 1024],
    'lr': [1e-2, 1e-3, 1e-4],
    'epochs': [50, 100, 150],
    'hidden_size': [20, 30, 40]
}


X_train_org, X_test_org, y_test_org = utils.data_generator(path=datafolder, dataset=dataset)
for name, clf in model_dict.items():
    if name != md_name:
        continue
    if name not in ['LSTM', 'LSTM_AE', 'LSTM_VAE']:
        if name == 'IForest':
            for params in _generate_param_combinations(param_grid1):
                n_estimators = params['n_estimators']
                max_samples = params['max_samples']
                contamination = params['contamination']
                md = clf(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination).fit(X_train_org)
                score_train = md.decision_function(X_train_org)
                score_test = md.decision_function(X_test_org)
                y_test = y_test_org.copy()
                info = f'_{n_estimators}_{max_samples}_{contamination}'
                write_ans(score_train, score_test, y_test, is_pub, dataset, name+info, save_path)

    elif name == 'LSTM':
        for params in _generate_param_combinations(param_grid2):
            batch_size = params['batch_size']
            lr = params['lr']
            epochs = params['epochs']
            hidden_size = params['hidden_size']
            X_train, X_train_next, _ = utils.data_split(X_train_org)
            train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(),
                                                    torch.from_numpy(X_train_next).float()),
                                      batch_size=batch_size, shuffle=True, drop_last=True)  # dataloader
            model = LSTM(input_size=X_train.shape[-1], hidden_size=hidden_size); print(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            fit_LSTM(train_loader=train_loader, model=model, optimizer=optimizer, epochs=epochs)

            X_test, X_test_next, y_test = utils.data_split(X_test_org, y_test_org)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(),
                                                   torch.from_numpy(X_test_next).float()),
                                     batch_size=batch_size, shuffle=False, drop_last=False)
            score_train = eval_LSTM(test_loader=train_loader, model=model)
            score_test = eval_LSTM(test_loader=test_loader, model=model)
            assert len(score_test) == len(y_test)
            info = f'_{batch_size}_{lr}_{epochs}_{hidden_size}'
            write_ans(score_train, score_test, y_test, is_pub, dataset, name+info, save_path)

    elif name in ['LSTM_AE', 'LSTM_VAE']:
        clf = LSTM_AE if name == 'LSTM_AE' else LSTM_VAE
        fit = fit_LSTM_AE if name == 'LSTM_AE' else fit_LSTM_VAE
        evl = eval_LSTM_AE if name == 'LSTM_AE' else eval_LSTM_VAE
        for params in _generate_param_combinations(param_grid):
            batch_size = params['batch_size']
            lr = params['lr']
            epochs = params['epochs']
            hidden_size = params['hidden_size']
            X_train, _, _ = utils.data_split(X_train_org)
            train_loader = DataLoader(torch.from_numpy(X_train).float(),
                                      batch_size=batch_size, shuffle=True, drop_last=True) # dataloader
            model = clf(input_size=X_train.shape[-1], hidden_size=hidden_size); print(model) # model
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer
            fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=epochs) # fitting

            X_test, _, y_test = utils.data_split(X_test_org, y_test_org)
            test_loader = DataLoader(torch.from_numpy(X_test).float(),
                                     batch_size=batch_size, shuffle=False, drop_last=False)
            score_train = evl(test_loader=train_loader, model=model)
            score_test = evl(test_loader=test_loader, model=model)
            assert len(score_test) == len(y_test)
            info = f'_{batch_size}_{lr}_{epochs}_{hidden_size}'
            write_ans(score_train, score_test, y_test, is_pub, dataset, name, save_path)

    else:
        raise NotImplementedError