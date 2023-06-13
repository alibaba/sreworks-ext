import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from myutils import Utils
from torch.utils.data import DataLoader, TensorDataset

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

utils = Utils()
path = "path"
dataset_list = ['instance name']

utils.set_seed(42)
utils.get_device()

model_dict = {
            'KNN': KNN, 'LOF': LOF,
              'IForest': IForest,
              'COPOD': COPOD, 'ECOD': ECOD,
              'DeepSVDD': DeepSVDD,
              'LSTM': LSTM, 'LSTM_AE': LSTM_AE, 'LSTM_VAE': LSTM_VAE} # 'OCSVM': OCSVM

batch_size = 512
lr = 1e-3
epochs = 50
hidden_size = 20

df_aucroc = pd.DataFrame(data=None, index=dataset_list, columns=model_dict.keys())
df_aucpr = pd.DataFrame(data=None, index=dataset_list, columns=model_dict.keys())

for dataset in tqdm(dataset_list):
    X_train_org, X_test_org, y_test_org = utils.data_generator(path=path, dataset=dataset)

    for name, clf in model_dict.items():

        if name not in ['LSTM', 'LSTM_AE', 'LSTM_VAE']:
            clf = clf().fit(X_train_org)
            score_train = clf.decision_function(X_train_org)
            score_test = clf.decision_function(X_test_org)
            y_test = y_test_org.copy()

        elif name == 'LSTM':
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

        elif name in ['LSTM_AE', 'LSTM_VAE']:
            clf = LSTM_AE if name == 'LSTM_AE' else LSTM_VAE
            fit = fit_LSTM_AE if name == 'LSTM_AE' else fit_LSTM_VAE
            evl = eval_LSTM_AE if name == 'LSTM_AE' else eval_LSTM_VAE

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

        else:
            raise NotImplementedError
        
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
        scores = combine_all_evaluation_scores(y_test, pred, score_test, dataset)
        scores['model'] = scores['model'] + '_' + name
        print(scores)
        with open('new_fillzero.csv','a',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=scores.keys())
            writer.writerow(scores)
        