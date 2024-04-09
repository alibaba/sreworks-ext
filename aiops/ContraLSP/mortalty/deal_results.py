import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
from sklearn import metrics
import pickle as pkl
import torch
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams.update({'font.size': 16})

from utils.metrics import get_entropy_array, get_information_array

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)
pd.set_option('max_colwidth', 500)

explainer_list = [
            "occlusion",
            "augmented_occlusion",
            "integrated_gradients",
            "gradient_shap",
            "deep_lift",
            "lime",
            "fit",
            "retain",
            "dyna_mask",
            "extremal_mask",  # tensor(13723.2715, grad_fn=<SumBackward0>) tensor(0.2366, grad_fn=<MeanBackward0>)
            "gate_mask",# tensor(14289.1562) tensor(0.4865, grad_fn=<MeanBackward0>) tensor(0.0310, gra>) 1.1 1 tensor(0.1030, grad_fn=<MseLossBackward0>)
        ]
baseline = "Average"    # Zeros, Average
Topk = 0.2
CV = 5
pd.options.display.float_format = '{:.3f}'.format
mymetrics = np.zeros((4, len(explainer_list), CV))
results_df = pd.DataFrame(columns=["Acc", "Astd", "CE", "CEstd", "Suff", "Sstd", "Comp", "Cstd"])
# Seed, Fold, Baseline, Topk, Explainer, Lambda_1, Lambda_2, Accuracy, Comprehensiveness, Cross
# Entropy, Log
# Odds, Sufficiency
data = pd.read_csv("./results.csv")

for cv in range(CV):
    for e, explainer in enumerate(explainer_list):
        mymetrics[0, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline].Accuracy.values[0]
        mymetrics[1, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline]['Cross Entropy'].values[0]
        mymetrics[2, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline]['Sufficiency'].values[0]
        mymetrics[3, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline].Comprehensiveness.values[0]
for e, explainer in enumerate(explainer_list):
    aup_avg, aup_std = np.mean(mymetrics[0, e, :]), np.std(mymetrics[0, e, :])
    aur_avg, aur_std = np.mean(mymetrics[1, e, :]), np.std(mymetrics[1, e, :])
    im_avg, im_std = np.mean(mymetrics[2, e, :])*100, np.std(mymetrics[2, e, :])*100
    sm_avg, sm_std = np.mean(mymetrics[3, e, :])*100, np.std(mymetrics[3, e, :])*100
    results_df.loc[explainer] = [aup_avg, aup_std, aur_avg, aur_std,
                                 im_avg, im_std, sm_avg, sm_std]

baseline = "Zeros"    # Zeros, Average
results_df2 = pd.DataFrame(columns=["Acc", "Astd", "CE", "CEstd", "Suff", "Sstd", "Comp", "Cstd"])
for cv in range(CV):
    for e, explainer in enumerate(explainer_list):
        mymetrics[0, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline].Accuracy.values[0]
        mymetrics[1, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline]['Cross Entropy'].values[0]
        mymetrics[2, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline]['Sufficiency'].values[0]
        mymetrics[3, e, cv] = data[data.Fold==cv][data.Topk==Topk][data.Explainer==explainer][data.Baseline==baseline].Comprehensiveness.values[0]
for e, explainer in enumerate(explainer_list):
    aup_avg, aup_std = np.mean(mymetrics[0, e, :]), np.std(mymetrics[0, e, :])
    aur_avg, aur_std = np.mean(mymetrics[1, e, :]), np.std(mymetrics[1, e, :])
    im_avg, im_std = np.mean(mymetrics[2, e, :])*100, np.std(mymetrics[2, e, :])*100
    sm_avg, sm_std = np.mean(mymetrics[3, e, :])*100, np.std(mymetrics[3, e, :])*100
    results_df2.loc[explainer] = [aup_avg, aup_std, aur_avg, aur_std,
                                 im_avg, im_std, sm_avg, sm_std]


resuldasdasdt = pd.concat([results_df, results_df2], axis=1)
print(resuldasdasdt)
############################################################################################
############################################################################################
############################################################################################
lab_name = {
            "occlusion":"FO",
            "fit":"FIT",
            "dynamask":"Dynamask",
    "gatemask":"ContraLSP",
    "extrmask":"Extrmask",
    "retain":"RETAIN",

}

explainer_list = [
            "occlusion",
            # "lime",
            "fit",
            "dyna_mask",
    "gate_mask",
    "extremal_mask",
    "retain",

]
baseline = "Average"
areas = [float(area) for area in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
metrics_array = np.zeros((len(explainer_list), len(areas), 4,  CV))
for cv in range(CV):
    for e, explainer in enumerate(explainer_list):
        for a, area in enumerate(areas):
            metrics_array[e, a, 0, cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline].Accuracy.values[0]
            metrics_array[e, a, 1,  cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline]['Cross Entropy'].values[0]
            metrics_array[e, a, 2, cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline]['Sufficiency'].values[0]
            metrics_array[e, a, 3, cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline].Comprehensiveness.values[0]


fig, axes = plt.subplots(1, 4, figsize=(19, 4))

# Plot the CE and the ACC for each attribution method and each mask area
for k, name in enumerate(explainer_list):
    # plt.figure(1)
    axes[0].plot(areas, metrics_array[k, :, 0, :].mean(axis=-1), label=name)
    axes[0].fill_between(
        areas,
        metrics_array[k, :, 0, :].mean(axis=-1) - metrics_array[k, :, 0, :].std(axis=-1),
        metrics_array[k, :, 0, :].mean(axis=-1) + metrics_array[k, :, 0, :].std(axis=-1),
        alpha=0.1,
    )
    # plt.figure(2)
    axes[1].plot(areas, metrics_array[k, :, 1, :].mean(axis=-1), label=name)
    axes[1].fill_between(
        areas,
        metrics_array[k, :, 1, :].mean(axis=-1) - metrics_array[k, :, 1, :].std(axis=-1),
        metrics_array[k, :, 1, :].mean(axis=-1) + metrics_array[k, :, 1, :].std(axis=-1),
        alpha=0.1,
    )
    # plt.figure(3)
    axes[2].plot(areas, metrics_array[k, :, 2, :].mean(axis=-1), label=name)
    axes[2].fill_between(
        areas,
        metrics_array[k, :, 2, :].mean(axis=-1) - metrics_array[k, :, 2, :].std(axis=-1),
        metrics_array[k, :, 2, :].mean(axis=-1) + metrics_array[k, :, 2, :].std(axis=-1),
        alpha=0.1,
    )
    # plt.figure(4)
    axes[3].plot(areas, metrics_array[k, :, 3, :].mean(axis=-1), label=name)
    axes[3].fill_between(
        areas,
        metrics_array[k, :, 3, :].mean(axis=-1) - metrics_array[k, :, 3, :].std(axis=-1),
        metrics_array[k, :, 3, :].mean(axis=-1) + metrics_array[k, :, 3, :].std(axis=-1),
        alpha=0.1,
    )
# plt.figure(1)
axes[0].set_ylabel("Accuracy", fontsize=18)
axes[1].set_ylabel("Cross Entropy", fontsize=18)
axes[2].set_ylabel("Sufficiency", fontsize=18)
axes[3].set_ylabel("Comprehensiveness", fontsize=18)

for i in range(4):
    axes[i].set_xlim(0.1, 0.6)
    axes[i].set_xlabel("Fraction of the average perturbed", fontsize=16)

# plt.tight_layout()
fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=0.35, hspace=0)
h, lab = axes[0].get_legend_handles_labels()
lab = [
            "occlusion",
            # "lime",
            "fit",
            "dynamask",
    "gatemask",
    "extrmask",
    "retain",

]
order = [0,1,5,2,4,3]

plt.legend([h[i] for i in order], [lab_name[lab[i]] for i in order], ncol = len(explainer_list), bbox_to_anchor=(0.16,1.3), fontsize=18)

plt.savefig("mimic_avg.pdf", bbox_inches="tight")



###########################################################################################
############################################################################################
############################################################################################
explainer_list = [
            "occlusion",
            # "lime",
            "fit",
            "dyna_mask",
    "gate_mask",
    "extremal_mask",
    "retain",

]
baseline = "Zeros"
areas = [float(area) for area in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
metrics_array = np.zeros((len(explainer_list), len(areas), 4,  CV))
for cv in range(CV):
    for e, explainer in enumerate(explainer_list):
        for a, area in enumerate(areas):
            metrics_array[e, a, 0, cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline].Accuracy.values[0]
            metrics_array[e, a, 1,  cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline]['Cross Entropy'].values[0]
            metrics_array[e, a, 2, cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline]['Sufficiency'].values[0]
            metrics_array[e, a, 3, cv] = data[data.Fold==cv][data.Topk==area][data.Explainer==explainer][data.Baseline==baseline].Comprehensiveness.values[0]


fig, axes = plt.subplots(1, 4, figsize=(19, 4))
# Plot the CE and the ACC for each attribution method and each mask area
for k, name in enumerate(explainer_list):
    # plt.figure(1)
    axes[0].plot(areas, metrics_array[k, :, 0, :].mean(axis=-1), label=name)
    axes[0].fill_between(
        areas,
        metrics_array[k, :, 0, :].mean(axis=-1) - metrics_array[k, :, 0, :].std(axis=-1),
        metrics_array[k, :, 0, :].mean(axis=-1) + metrics_array[k, :, 0, :].std(axis=-1),
        alpha=0.1,
    )
    # plt.figure(2)
    axes[1].plot(areas, metrics_array[k, :, 1, :].mean(axis=-1), label=name)
    axes[1].fill_between(
        areas,
        metrics_array[k, :, 1, :].mean(axis=-1) - metrics_array[k, :, 1, :].std(axis=-1),
        metrics_array[k, :, 1, :].mean(axis=-1) + metrics_array[k, :, 1, :].std(axis=-1),
        alpha=0.1,
    )
    # plt.figure(3)
    axes[2].plot(areas, metrics_array[k, :, 2, :].mean(axis=-1), label=name)
    axes[2].fill_between(
        areas,
        metrics_array[k, :, 2, :].mean(axis=-1) - metrics_array[k, :, 2, :].std(axis=-1),
        metrics_array[k, :, 2, :].mean(axis=-1) + metrics_array[k, :, 2, :].std(axis=-1),
        alpha=0.1,
    )
    # plt.figure(4)
    axes[3].plot(areas, metrics_array[k, :, 3, :].mean(axis=-1), label=name)
    axes[3].fill_between(
        areas,
        metrics_array[k, :, 3, :].mean(axis=-1) - metrics_array[k, :, 3, :].std(axis=-1),
        metrics_array[k, :, 3, :].mean(axis=-1) + metrics_array[k, :, 3, :].std(axis=-1),
        alpha=0.1,
    )
# plt.figure(1)
axes[0].set_ylabel("Accuracy", fontsize=18)
axes[1].set_ylabel("Cross Entropy", fontsize=18)
axes[2].set_ylabel("Sufficiency", fontsize=18)
axes[3].set_ylabel("Comprehensiveness", fontsize=18)

for i in range(4):
    axes[i].set_xlim(0.1, 0.6)
    axes[i].set_xlabel("Fraction of the zero perturbed", fontsize=16)

fig.tight_layout()
plt.subplots_adjust(wspace=0.35, hspace=0)
h, lab = axes[0].get_legend_handles_labels()
lab = [
            "occlusion",
            # "lime",
            "fit",
            "dynamask",
    "gatemask",
    "extrmask",
    "retain",
]
order = [0,1,5,2,4,3]

plt.legend([h[i] for i in order], [lab_name[lab[i]] for i in order], ncol = len(explainer_list), bbox_to_anchor=(0.16,1.3), fontsize=18)

plt.savefig("mimic_zeros.pdf", bbox_inches="tight")




print(resuldasdasdt)


