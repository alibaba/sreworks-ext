import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
from sklearn import metrics
import pickle as pkl
import torch
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from utils.metrics import get_entropy_array, get_information_array

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 500)


explainer_list = [0.1, 0.5 ,1., 2., 5.]#
CV = 5
path = "./lambda_study.csv"

pd.options.display.float_format = '{:.2f}'.format
metrics = np.zeros((4, len(explainer_list), len(explainer_list), CV))
results_df = pd.DataFrame(columns=["AUP_01", "AUPstd_01", "AUR_01", "AURstd_01",
"AUP_05", "AUPstd_05", "AUR_05", "AURstd_05",
"AUP_1", "AUPstd_1", "AUR_1", "AURstd_1",
"AUP_2", "AUPstd_2", "AUR_2", "AURstd_2",
"AUP_5", "AUPstd_5", "AUR_5", "AURstd_5"
                                   ])
data = pd.read_csv(path)
data = data[data.Explainer=='gate_mask']
for cv in range(CV):
    for e, l1 in enumerate(explainer_list):
        for f, l2 in enumerate(explainer_list):
            metrics[0, e, f, cv] = data[data.Fold==cv][data.Lambda_1==l1][data.Lambda_2==l2].AUP.values[0]
            metrics[1, e, f, cv] = data[data.Fold==cv][data.Lambda_1==l1][data.Lambda_2==l2].AUR.values[0]
            metrics[2, e, f, cv] = data[data.Fold==cv][data.Lambda_1==l1][data.Lambda_2==l2].Information.values[0]
            metrics[3, e, f, cv] = data[data.Fold==cv][data.Lambda_1==l1][data.Lambda_2==l2].Entropy.values[0]

for f, l2 in enumerate(explainer_list):
    one_line = []
    for e, l1 in enumerate(explainer_list):
        aup_avg, aup_std = np.mean(metrics[0, e, f, :]), np.std(metrics[0, e, f, :])
        aur_avg, aur_std = np.mean(metrics[1, e, f, :]), np.std(metrics[1, e, f, :])
        one_line = one_line+[aup_avg, aup_std, aur_avg, aur_std]
    results_df.loc[str(l2)] = one_line
print(path)
print(results_df)
