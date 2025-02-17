import torch
from torch.utils import data


from model.modules.QueryFormer.utils import *

    
class Tensor_Opt_modal_dataset(data.Dataset):
    '''
        sample: [feature, label]
    '''
    def __init__(self, df, device, train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = encoding
        self.treeNodes = []

        samples_list = df.values.tolist()
        samples_data = []

        if train_dataset is None:
            # 训练集归一化操作
            logs = []
            timeseries = []
            opt_labels = []
            querys = []
            for i, samples in enumerate(samples_list):
                querys.append(samples[0])
                logs.append(torch.tensor(samples[2]))
                timeseries.append(torch.tensor(samples[3]))
                opt_labels.append(torch.tensor(samples[5]))
            

            logs = torch.stack(logs, dim=0)
            self.logs_train_mean = logs.mean(dim=0)
            self.logs_train_std = logs.std(dim=0)

            timeseries = torch.stack(timeseries, dim=0)
            self.timeseries_train_mean = timeseries.mean(dim=[0, 2])
            self.timeseries_train_std = timeseries.std(dim=[0, 2])

            opt_labels = torch.stack(opt_labels, dim=0)
            self.opt_labels_train_mean = opt_labels.mean(dim=0)
            self.opt_labels_train_std = opt_labels.std(dim=0)

        else:
            querys = df["query"].values.tolist()

            self.logs_train_mean = train_dataset.logs_train_mean
            self.logs_train_std = train_dataset.logs_train_std
            self.timeseries_train_mean = train_dataset.timeseries_train_mean
            self.timeseries_train_std = train_dataset.timeseries_train_std
            self.opt_labels_train_mean = train_dataset.opt_labels_train_mean
            self.opt_labels_train_std = train_dataset.opt_labels_train_std

        for i, samples in enumerate(samples_list):
            sam = {
                    "query": samples[0], 
                   "plan": samples[1], 
                   "log": (torch.tensor(samples[2]) - self.logs_train_mean) / (self.logs_train_std + 1e-6),
                    "timeseries": (torch.tensor(samples[3]) - self.timeseries_train_mean.unsqueeze(1)) / (self.timeseries_train_std.unsqueeze(1) + 1e-6), 
                    "multilabel": torch.tensor(samples[4]), 
                    "opt_label": (torch.tensor(samples[5])  - self.opt_labels_train_mean) / (self.opt_labels_train_std + 1e-6),
                    "duration": samples[6],
                    "ori_opt_label": (torch.tensor(samples[5])  - self.opt_labels_train_mean) / (self.opt_labels_train_std + 1e-6)
            }
            samples_data.append(sam)

        self.samples = samples_data
        self.device = device
        self.train = train

    def __getitem__(self,index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    
