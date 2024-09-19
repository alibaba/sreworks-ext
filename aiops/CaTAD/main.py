#%%
import os
import csv
import random
import argparse
import pandas as pd
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.CaTAD import CaTAD
from torch.nn.utils import clip_grad_value_
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from utils.helper import orthogonal_loss, str_to_bool
import logging
from utils.logging import get_logger


parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str, default='./data/', help='Location of datasets.')
parser.add_argument('--dataset', type=str, default='SMD', help='The dataset name.')            
parser.add_argument('--output_dir', type=str, default='./checkpoint/')
parser.add_argument('--name',default='CaTAD')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--log_dir',default='./log/')
parser.add_argument('--seed_list', type=list, default=[2018,2019,2020,2021,2022], help='Random seed to use.')

#
parser.add_argument('--use_multidim', type=str_to_bool, default=True)

# parameters
parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks PeNF to stack.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers.')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch_norm', type=bool, default=False)

parser.add_argument('--alpha', type=float, default=0.01, help = 'Balance the loss')
parser.add_argument('--beta', type=float, default=0.01, help = 'Balance the loss')
parser.add_argument('--interve_level', type=float, default=1, help = 'The degree we do the causal intervetion.')

# training params
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

name = '{}_{}'.format(args.name, args.dataset)
save_path = os.path.join(args.output_dir, args.name, args.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)

### log
log_path = os.path.join(args.log_dir, args.name)
logger = get_logger(log_path, name, 'info_{}.log'.format(args.dataset), level=logging.INFO)
logger.info(args)
            
### dataset
from utils.helper import get_loader
train_loader, val_loader, test_loader, n_sensor, period, unit = get_loader(args)

roc_test_list, y_score_list, y_label_list, all_score_list, all_label_list=[], [], [], [], []
training_time=[]

for seed in args.seed_list:
    ### set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    epoch = 0
    model = CaTAD(args.n_blocks, 1, args.hidden_size, args.n_layers, dropout=args.dropout, 
                    batch_norm=args.batch_norm, n_node=n_sensor, interve_level=args.interve_level, period=period, unit=unit, use_multidim=args.use_multidim)
    model = model.to(device)

    loss_best = 100
    a_ctrl = 0.00001 # unsure the loss in the same magnitude

    optimizer = torch.optim.Adam([
        {'params':model.parameters(), 'weight_decay':args.weight_decay}], lr=args.lr , weight_decay=args.weight_decay)
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    if args.mode == 'train':
        for _ in range(args.n_epochs):
            start_time = time.time()
            loss_train = []
            epoch += 1
            model.train()
            for x, x_mark in train_loader:
                x = x.to(device)
                x_mark = x_mark.to(device)

                optimizer.zero_grad()
                loss, cause, x_enc, x_aug = model(x, x_mark)

                total_loss = (1-args.alpha-args.beta)*loss + args.alpha*orthogonal_loss(cause)*a_ctrl + args.beta * cosine_sim(x_enc, x_aug).mean()
                total_loss.backward()
                clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                loss_train.append(loss.item())
            end_time = time.time()

            # eval 
            model.eval()
            loss_val = []
            with torch.no_grad():
                for x, x_mark in val_loader:
                    x = x.to(device)
                    x_mark = x_mark.to(device)
                    loss, _, _, _ = model.test(x,x_mark)
                    loss_val.append(loss.cpu().numpy())
            loss_val = np.concatenate(loss_val)

            loss_test = []
            with torch.no_grad():
                for x, x_mark in test_loader:
                    x = x.to(device)
                    x_mark = x_mark.to(device)
                    loss, _, _, _ = model.test(x,x_mark)
                    loss_test.append(loss.cpu().numpy())
            loss_test = np.concatenate(loss_test)

            loss_val = np.nan_to_num(loss_val)
            loss_test = np.nan_to_num(loss_test)

            roc_val = roc_auc_score(np.asarray(val_loader.dataset.label.values,dtype=int),loss_val)
            roc_test = roc_auc_score(np.asarray(test_loader.dataset.label.values,dtype=int),loss_test)
            logger.info('Epoch: {}, train -log_prob: {:.2f}, test -log_prob: {:.2f}, roc_val: {:.4f}, roc_test: {:.4f}, Training time: {}'.format(epoch, np.mean(loss_train), np.mean(loss_val), roc_val, roc_test, (end_time - start_time)))
            training_time.append((end_time - start_time))

            if np.mean(loss_val) < loss_best:
                loss_best = np.mean(loss_val)
                print("save model {} epoch".format(epoch))
                torch.save(model.state_dict(), os.path.join(save_path, "{}_best_seed{}.pt".format(args.name, seed)))

    logger.info('======== Evaluation ========')
    model.load_state_dict(torch.load(os.path.join(save_path, "{}_best_seed{}.pt".format(args.name, seed))))
    model.eval()

    loss_test = []
    with torch.no_grad():
        for x, x_mark in test_loader:

            x = x.to(device)
            x_mark = x_mark.to(device)
            loss, _, _, _ = model.test(x, x_mark)
            loss_test.append(loss.cpu().numpy())
    loss_test = np.concatenate(loss_test)
    roc_test = roc_auc_score(np.asarray(test_loader.dataset.label.values,dtype=int),loss_test)
    logger.info("The ROC score on {} dataset is {}".format(args.dataset, roc_test))
    roc_test_list.append(roc_test)
    
logger.info('After running experiments for seed {}, the ROC results are {} ± {}, Training time per epoch {}'.format(args.seed_list, np.round(np.mean(roc_test_list),3 ), np.round(np.std(roc_test_list), 3), np.round(np.mean(training_time), 3)))

csv_path = os.path.join(args.log_dir, args.name, 'results_{}.csv'.format(args.dataset))
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns = ['time', 'seed','n_blocks','hidden_size','n_layers','dropout','alpha', 'beta','interve_level', 'n_epochs', 'roc_mean', 'roc_std', 'training_time'])
    df.to_csv(csv_path, index = False)

with open(csv_path,'a+') as f:
    csv_write = csv.writer(f)
    data_row = [time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), # time
                args.seed_list, # seed
                args.n_blocks, 
                args.hidden_size,
                args.n_layers,
                args.dropout,
                args.alpha,
                args.beta,
                args.interve_level,
                args.n_epochs,
                np.round(np.mean(roc_test_list),3 ),
                np.round(np.std(roc_test_list), 3),
                np.round(np.mean(training_time),3),
                ]
    csv_write.writerow(data_row)