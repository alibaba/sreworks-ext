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


def process_results_by_file(CV, explainer_list, path='./results.csv'):
    pd.options.display.float_format = '{:.2f}'.format
    metrics = np.zeros((4, len(explainer_list), CV))
    results_df = pd.DataFrame(columns=["AUP", "AUP std", "AUR", "AUR std", "Info", "Info std", "Entr", "Entr std"])
    data = pd.read_csv(path)
    for cv in range(CV):
        for e, explainer in enumerate(explainer_list):
            metrics[0, e, cv] = data[data.Fold==cv][data.Explainer==explainer].AUP.values[0]
            metrics[1, e, cv] = data[data.Fold==cv][data.Explainer==explainer].AUR.values[0]
            metrics[2, e, cv] = data[data.Fold==cv][data.Explainer==explainer].Information.values[0]
            metrics[3, e, cv] = data[data.Fold==cv][data.Explainer==explainer].Entropy.values[0]
    for e, explainer in enumerate(explainer_list):
        aup_avg, aup_std = np.mean(metrics[0, e, :]), np.std(metrics[0, e, :])
        aur_avg, aur_std = np.mean(metrics[1, e, :]), np.std(metrics[1, e, :])
        im_avg, im_std = np.mean(metrics[2, e, :])/10000, np.std(metrics[2, e, :])/10000
        sm_avg, sm_std = np.mean(metrics[3, e, :])/1000, np.std(metrics[3, e, :])/1000
        results_df.loc[explainer] = [aup_avg, aup_std, aur_avg, aur_std,
                                     im_avg, im_std, sm_avg, sm_std]
    print(path)
    print(results_df)


def process_results(CV, explainer_list, path="experiments/results/rare_time"):
    pd.options.display.float_format = '{:.2f}'.format
    metrics = np.zeros((6, len(explainer_list), CV))
    results_df = pd.DataFrame(columns=["AUP", "AUP std", "AUR", "AUR std", "AUROC", "AUROC std",
                                       "AUPRC", "AUPRC std", "Info", "Info std", "Entr", "Entr std"])
    for cv in range(CV):
        with open(path + f"true_saliency_{cv}.pkl", "rb") as f:
            true_saliency = pkl.load(f).cpu().numpy()
        for e, explainer in enumerate(explainer_list):
            with open(path + f"{explainer}_saliency_{cv}.pkl", "rb") as f:
                pred_saliency = pkl.load(f)
                if torch.is_tensor(pred_saliency):
                    pred_saliency = pred_saliency.cpu().numpy()
            prec, rec, thres = precision_recall_curve(true_saliency.flatten(), pred_saliency.flatten())
            metrics[0, e, cv] = auc(thres, prec[1:])
            metrics[1, e, cv] = auc(thres, rec[1:])
            auc_score = roc_auc_score(true_saliency.flatten(), pred_saliency.flatten())
            metrics[2, e, cv] = auc_score
            auprc_score = auc(rec, prec) if rec.shape[0] > 1 else -1
            metrics[3, e, cv] = auprc_score

            # Normalize the saliency map:
            pred_saliency -= pred_saliency.min(axis=(1, 2), keepdims=True)
            pred_saliency /= pred_saliency.max(axis=(1, 2), keepdims=True)
            sub_saliency = pred_saliency[true_saliency != 0]  # This is the saliency scores for each truly salient input
            metrics[4, e, cv] = get_information_array(sub_saliency, eps=1.0e-5)
            metrics[5, e, cv] = get_entropy_array(sub_saliency, eps=1.0e-5)

    for e, explainer in enumerate(explainer_list):
        aup_avg, aup_std = np.mean(metrics[0, e, :]), np.std(metrics[0, e, :])
        aur_avg, aur_std = np.mean(metrics[1, e, :]), np.std(metrics[1, e, :])
        auroc_avg, auroc_std = np.mean(metrics[2, e, :]), np.std(metrics[2, e, :])
        auprc_avg, auprc_std = np.mean(metrics[3, e, :]), np.std(metrics[3, e, :])
        im_avg, im_std = np.mean(metrics[4, e, :])/10000, np.std(metrics[4, e, :])/10000
        sm_avg, sm_std = np.mean(metrics[5, e, :])/100, np.std(metrics[5, e, :])/100
        results_df.loc[explainer] = [aup_avg, aup_std, aur_avg, aur_std,
                                     auroc_avg, auroc_std, auprc_avg, auprc_std,
                                     im_avg, im_std, sm_avg, sm_std]

    print(path)
    print(results_df)


def print_results(mask_label, true_label):
    if torch.is_tensor(mask_label):
        mask_label = mask_label.cpu().numpy()
    if torch.is_tensor(true_label):
        true_label = true_label.cpu().numpy()
    mask_prec, mask_rec, mask_thres = metrics.precision_recall_curve(
        true_label.flatten().astype(int), mask_label.flatten())
    print(f"Saliency AUROC: {metrics.roc_auc_score(true_label.flatten(), mask_label.flatten())}")
    print(f"Saliency AUPRC: {metrics.auc(mask_rec, mask_prec)}")
    print(f"Saliency AUP: {metrics.auc(mask_thres, mask_prec[:-1])}")
    print(f"Saliency AUR: {metrics.auc(mask_thres, mask_rec[:-1])}")
    return metrics


def plot_example_box(input_arrays, cur_id=0, save_location=None, k=8):
    sns.set()

    fig, ax = plt.subplots()
    plt.axis("off")
    input_array = input_arrays[cur_id].T
    # color_map = sns.diverging_palette(10, 133, as_cmap=True)
    # color_map = cm.Blues
    # norm = colors.BoundaryNorm([0,0.5,1], color_map.N)

    cmap1 = cm.get_cmap('YlGn', k)
    cmap1 = cmap1(np.linspace(0, 1, k))
    from matplotlib.colors import ListedColormap
    new_cmap = ListedColormap(cmap1)

    # sns.heatmap(data=input_array, cmap=color_map, cbar_kws={"label": "Mask"}, vmin=0, vmax=1)
    ax.imshow(input_array, interpolation="nearest",cmap=new_cmap)#"gray" , norm=norm
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.figure(figsize=(12, 2))
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if save_location:
        plt.savefig(str(save_location), bbox_inches="tight", pad_inches=0,dpi=600)
    else:
        plt.show()

    plt.close()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss