import torch

import pandas as pd
from transformers import BertTokenizer

from utils.data_tensor import Tensor_Opt_modal_dataset
from model.modules.QueryFormer.utils import Encoding


def load_dataset(data_path, batch_size = 8, device="cpu"):
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    path = data_path
    print("data path", path)
    
    df = pd.read_pickle(path)

    encoding = Encoding(None, {'NA': 0})

    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    print(df_test.shape)

    train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset), train_dataset 


def load_dataset_valid(data_path, batch_size = 8, device="cpu"):
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    path = data_path
    
    df = pd.read_pickle(path)

    encoding = Encoding(None, {'NA': 0})

    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    df_valid = df[df["dataset_cls"] == "valid"]

    train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    
    valid_dataset = Tensor_Opt_modal_dataset(df_valid[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, valid_dataloader, len(train_dataset), len(test_dataset), len(valid_dataset), train_dataset 


