import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import os
import csv
import ast
from utils.helper import FFT_for_Period

class CaTAD_Data(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10, unit='s'):
        super(CaTAD_Data, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.unit=unit
        self.data, self.idx, self.label, self.ts = self.preprocess(df,label)

    def preprocess(self, df, label):
        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit=self.unit)

        months = pd.Series(df.index).dt.month
        days = pd.Series(df.index).dt.day
        weekdays = pd.Series(df.index).dt.weekday  
        hours = pd.Series(df.index).dt.hour
        minutes = pd.Series(df.index).dt.minute
        if self.unit == 's':
            seconds = pd.Series(df.index).dt.second
            time_features = np.column_stack((months, days, weekdays, hours, minutes, seconds))
        else:
            time_features = np.column_stack((months, days, weekdays, hours, minutes))
        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]], time_features

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])
        data_mark = self.ts[start:end]
        return torch.FloatTensor(data).transpose(0,1), torch.Tensor(data_mark)

def load_wadi(root, batch_size, model):
    data = pd.read_csv(root, skiprows=1).iloc[:-2,:]
    data.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)':'label'},inplace=True)
    data.loc[data['label'] == 1, 'label'] = 0
    data.loc[data['label'] == -1, 'label'] = 1
    data["Timestamp"] = pd.date_range(start=data.loc[0, 'Date '], end=data.loc[len(data)-1, 'Date '] , freq='s')
    data.drop(columns=['Row ', 'Date ', 'Time', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS'], inplace=True) # Drop the empty columns
    data = data.set_index("Timestamp")

    print('Train {}, val {}, test {}'.format(int(0.6*len(data)), int(0.2*len(data)), (len(data)-int(0.8*len(data)))))
    print('N_dim {}'.format(data.shape[1]-1))
    print('Anomaly ratio: {}'.format((np.sum(data['label'] == 1)/len(data))*100))

    feature = data.iloc[:,:-1]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)
    period = FFT_for_Period(torch.Tensor(norm_feature.values).unsqueeze(0), None,'global')

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]
    
    train_loader = DataLoader(CaTAD_Data(train_df,train_label, unit='s'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CaTAD_Data(val_df,val_label, unit='s'), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CaTAD_Data(test_df,test_label, unit='s'), batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader, n_sensor, period, 's'

def load_psm(root, batch_size, model):
    data = pd.read_csv(root+"test.csv")
    data["timestamp_(min)"] = pd.date_range(start='1970-01-01', periods= len(data), freq='min')
    data = data.rename(columns={"timestamp_(min)": "Timestamp"})
    data = data.set_index("Timestamp")
    labels = pd.read_csv(root+"test_label.csv").iloc[:,1].values
    data['label'] = labels

    print('Train {}, val {}, test {}'.format(int(0.6*len(data)), int(0.2*len(data)), (len(data)-int(0.8*len(data)))))
    print('N_dim {}'.format(data.shape[1]))
    print('Anomaly ratio: {}'.format((np.sum(labels)/len(data))*100))

    feature = data.iloc[:,:-1]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)
    period = FFT_for_Period(torch.Tensor(norm_feature.values).unsqueeze(0), None,'global')

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]

    train_loader = DataLoader(CaTAD_Data(train_df,train_label, unit='min'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CaTAD_Data(val_df,val_label, unit='min'), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CaTAD_Data(test_df,test_label, unit='min'), batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader, n_sensor, period, 'min'

def load_smd(root, batch_size, entity, model):
    data = np.genfromtxt(os.path.join(root , 'test', '{}.txt'.format(entity)), dtype=np.float32, delimiter=',')
    label = np.genfromtxt(os.path.join(root, "test_label", '{}.txt'.format(entity)), dtype=np.float32,delimiter=',')
    label = pd.DataFrame(label, columns=["label"])
    header_list = ["col_%d" % i for i in range(data.shape[1])]
    data = pd.DataFrame(data, columns=header_list)
    data["Timestamp"] = pd.date_range(start='1970-01-01', periods= len(data), freq='min')
    data["label"] = np.array(label.values, dtype=int)
    data = data.set_index("Timestamp")

    print('Train {}, val {}, test {}'.format(int(0.6*len(data)), int(0.2*len(data)), (len(data)-int(0.8*len(data)))))
    print('N_dim {}'.format(data.shape[1]-1))
    print('Anomaly ratio: {}'.format((np.sum(label.values)/len(data))*100))

    feature = data.iloc[:,:-1]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)
    period = FFT_for_Period(torch.Tensor(norm_feature.values).unsqueeze(0), None,'global')

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]


    train_loader = DataLoader(CaTAD_Data(train_df,train_label, unit='min'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CaTAD_Data(val_df,val_label, unit='min'), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CaTAD_Data(test_df,test_label, unit='min'), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor, period, 'min'

def load_msl(root, batch_size, dataset, model):
    with open(os.path.join(root, 'labeled_anomalies.csv'), 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])
    data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']

    data, labels = [], []
    for row in data_info:
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=int)
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = 1
        labels.extend(label)
        filename = row[0]
        temp = np.load(os.path.join(root, 'test', filename + '.npy'))
        data.extend(temp)
    data = np.asarray(data)

    header_list = ["col_%d" % i for i in range(data.shape[1])]
    data = pd.DataFrame(data, columns=header_list)
    data["Timestamp"] = pd.date_range(start='1970-01-01', periods= len(data), freq='min')
    data["label"] = np.asarray(labels, dtype=int)
    data = data.set_index("Timestamp")

    print('Train {}, val {}, test {}'.format(int(0.6*len(data)), int(0.2*len(data)), (len(data)-int(0.8*len(data)))))
    print('N_dim {}'.format(data.shape[1]-1))
    print('Anomaly ratio: {}'.format((np.sum(data["label"].values)/len(data))*100))

    feature = data.iloc[:,:-1]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)
    period = FFT_for_Period(torch.Tensor(norm_feature.values).unsqueeze(0), None,'global')

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]


    train_loader = DataLoader(CaTAD_Data(train_df,train_label, unit='min'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CaTAD_Data(val_df,val_label, unit='min'), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CaTAD_Data(test_df,test_label, unit='min'), batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader, n_sensor, period, 'min'


