import os
import numpy as np
import pandas as pd
import random
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

class Utils():
    def __init__(self):
        pass

    def set_seed(self, seed):
        # basic seed
        np.random.seed(seed)
        random.seed(seed)

        # pytorch seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_device(self, gpu_specific=True):
        if gpu_specific:
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                print(f'number of gpu: {n_gpu}')
                print(f'cuda name: {torch.cuda.get_device_name(0)}')
                print('GPU is on')
            else:
                print('GPU is off')

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    def rm_na(self, X, y=None):
        idx = np.isnan(X).any(axis=1)
        idx = ~idx
        X = X[idx]
        if y is not None:
            y = y[idx]

        return X, y

    def metric(self, y_true, y_score, pos_label=1):
        aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
        aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

        return {'aucroc': aucroc, 'aucpr': aucpr}

    def data_generator(self, path, dataset, normalize=False):
        try:
            X_train = np.load(os.path.join(path, dataset, 'train.npy'), allow_pickle=True)
            X_test = np.load(os.path.join(path, dataset, 'test.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(path, dataset, 'test_label.npy'), allow_pickle=True)
        except:
            X_train = pd.read_csv(os.path.join(path, dataset, 'train.csv'))
            X_test = pd.read_csv(os.path.join(path, dataset, 'test.csv'))
            y_test = pd.read_csv(os.path.join(path, dataset, 'test_label.csv'))

            X_train = X_train.iloc[:, 1:].values
            X_test = X_test.iloc[:, 1:].values
            y_test = y_test.iloc[:, 1].values

        print(f'The minimum value of X_train: {np.min(X_train)}, X_test: {np.min(X_test)}')
        print(f'The maximum value of X_train: {np.max(X_train)}, X_test: {np.max(X_test)}')

        print(f'The shape of X_train: {X_train.shape}, the shape of X_test: {X_test.shape}')
        X_train, _ = self.rm_na(X_train)
        X_test, y_test = self.rm_na(X_test, y_test)

        print(f'The shape of X_train (after removing nan): {X_train.shape}, the shape of X_test: {X_test.shape}\n')
        assert X_test.shape[0] == len(y_test)

        if normalize:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return X_train, X_test, y_test

    def data_split(self, X, y=None, timestep=5):
        sequences, sequences_next = [], []
        for i in range(X.shape[0]-timestep):
            seq = X[i:i+timestep]
            seq_next = X[i+timestep]
            sequences.append(seq)
            sequences_next.append(seq_next)
        sequences = np.stack(sequences)
        sequences_next = np.stack(sequences_next)

        assert sequences.shape[0] == sequences_next.shape[0]

        if y is not None:
            y = y[timestep:]
            assert sequences.shape[0] == len(y)

        return sequences, sequences_next, y


