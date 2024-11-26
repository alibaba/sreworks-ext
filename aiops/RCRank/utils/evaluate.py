import torch

import pandas as pd
import scipy.stats as stats


def extended_tau_2(list_a, list_b, all_label):
    if len(list_a) < len(list_b):
        for i in range(len(list_b) - len(list_a)):
            list_a.append((set(all_label) - set(list_a) - set(list_b)).pop())
    if len(list_a) == 0 and len(list_a) == len(list_b):
        return 1.0
    if len(list_b) == 0:
        return 0.0
    ranks = join_ranks(create_rank(list_a), create_rank(list_b)).fillna(12)
    dummy_df = pd.DataFrame([{'rank_a': 12, 'rank_b': 12} for i in range(2*len(list_a)-len(ranks))])
    total_df = pd.concat([ranks, dummy_df])
    return scale_tau(len(list_a), stats.kendalltau(total_df['rank_a'], total_df['rank_b'])[0])

def scale_tau(length, value):
    n_0 = 2*length*(2*length-1)
    n_a = length*(length-1)
    n_d = n_0 - n_a
    min_tau = (2.*n_a - n_0) / (n_d)
    return 2*(value-min_tau)/(1-min_tau) - 1

def create_rank(a):
    return pd.DataFrame(
                zip(a, range(len(a))),
                columns=['key', 'rank'])\
            .set_index('key')

def join_ranks(rank_a, rank_b):
    return rank_a.join(rank_b, lsuffix='_a', rsuffix='_b', how='outer')

def evaluate_tau(label_list, pred_list):
    tau_res = []
    all_label = list(range(12))
    for i in range(len(label_list)):
        tau_res.append(extended_tau_2(label_list[i], pred_list[i], all_label))
    return torch.tensor(tau_res).mean().item()

