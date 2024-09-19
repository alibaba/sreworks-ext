import torch
import pandas as pd
import numpy as np
from sklearn.metrics import auc
import os
from datetime import datetime

def get_timestamp(stamps):
    return (stamps - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

def roc_auc(label_time, pred, negative_sample, sigma):
    negative_sample = np.sort(negative_sample)[::-1]
    thresholds = list(negative_sample[::int(len(negative_sample)/50)])
    thresholds.append(negative_sample[-1])
    tps=[]
    fps=[]

    for thre in thresholds:
        pred_pos = pred[pred>thre]

        tp = 0
        for i in range(len(label_time)):
            start_time = label_time[i] - pd.Timedelta(30, unit='min')
            end_time = label_time[i] + pd.Timedelta(30, unit='min')

            detected_event = pred_pos[str(start_time): str(end_time)]
            if len(detected_event)>0:
                timestamps = get_timestamp(detected_event.index)
                delta_t = np.min(np.abs(timestamps.values - get_timestamp(label_time[i])))
                tp += np.exp(-np.power(delta_t/sigma,2))
        tp = tp/len(label_time)
        tps.append(tp)

        fp = (negative_sample>thre).sum()/len(negative_sample)
        fps.append(fp)
    return auc(fps,tps), (fps,tps)

def roc_auc_all(loss_np, delta_t, sigma):

    ground_truth = np.exp(-np.power((delta_t.values)/sigma,2))

    loss_sort = np.sort(loss_np)[::-1]
    thresholds = list(loss_sort[::int(len(loss_sort)/50)])
    thresholds.append(loss_sort[-1])

    n_pos = ground_truth.sum()
    n_neg = (1-ground_truth).sum()
    tps = []
    fps = []
    for thre in thresholds:
        pred_pos = loss_np>thre

        tp = ground_truth[pred_pos].sum()/n_pos
        fp = (1-ground_truth[pred_pos]).sum()/n_neg
        tps.append(tp)
        fps.append(fp)

    auc_score = auc(fps, tps)
    return auc_score, fps, tps


def orthogonal_loss(h):
    """
    Compute the loss term to encourage orthogonality in the hidden state vectors.
    h: (batch_size, num_vectors, vector_dim) Tensor of hidden state vectors.
    """
    h_norm = h / (h.norm(p=2, dim=-1, keepdim=True) + 1e-6)
    dot_product = torch.matmul(h_norm, h_norm.transpose(-1, -2))
    identity = torch.eye(dot_product.size(-1), device=dot_product.device)
    dot_product -= identity * dot_product.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    loss = (dot_product ** 2).sum()
    return loss

def high_freq_aug(ts, interve_level):
    bs, t, d = ts.shape
    ts_fft = torch.fft.fft(ts, dim=1)

    high_freq_threshold = int(t * 0.25) # we set high frequency threshold are 25%
    num_high_freq = t - 2 * high_freq_threshold

    noise = interve_level * torch.randn(bs, num_high_freq, d).cuda()
    ts_fft[:, high_freq_threshold:-high_freq_threshold, :] += noise
    ts_aug = torch.fft.ifft(ts_fft, dim=1)
    return ts_aug.real.type(torch.float32)

def causal_interve(x, interve_level):
    return high_freq_aug(x, interve_level)

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_loader(args):
    print("Loading dataset....")
    if args.dataset == 'WADI':
        from utils.dataset import load_wadi
        train_loader, val_loader, test_loader, n_sensor, period, unit = load_wadi(args.data_dir + "WADI_attackdataLABLE.csv", args.batch_size, args.name)
    elif args.dataset == 'PSM':
        from utils.dataset import load_psm
        train_loader, val_loader, test_loader, n_sensor, period, unit = load_psm(args.data_dir + "PSM/", args.batch_size, args.name)
    elif args.dataset == 'SMD':
        from utils.dataset import load_smd
        train_loader, val_loader, test_loader, n_sensor, period, unit = load_smd(args.data_dir + "SMD/", args.batch_size, entity='machine-1-4', model=args.name)
    elif args.dataset == 'MSL':
        from utils.dataset import load_msl
        train_loader, val_loader, test_loader, n_sensor, period, unit = load_msl(args.data_dir + "MSL/", args.batch_size, dataset='MSL', model=args.name)

    return train_loader, val_loader, test_loader, n_sensor, period, unit

def FFT_for_Period(x, k=2, method='local', window_size = 60):
    '''
    refer to Time-Series-Library https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
    '''
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    if method == 'local':
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]
    elif method == 'global':
        frequency_list = frequency_list.detach().cpu().numpy()
        period = x.shape[1] // max(frequency_list)
        assert period < window_size
        return int(period)
