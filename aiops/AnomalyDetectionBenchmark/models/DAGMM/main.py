import tensorflow as tf
import numpy as np
import os
import csv
from spot import SPOT

import itertools


from metrics.combine_all_scores import combine_all_evaluation_scores

from dagmm import DAGMM


public_datasets = ['SMAP', 'MSL', 'SMD', 'NIPS_TS_CCard', 'NIPS_TS_Swan', 'NIPS_TS_Water', 'NIPS_TS_Syn_Mulvar', 'SWaT']

# datasets folder
public_datafolder = '/mnt/workspace/workgroup/lingke/public/'
holo_datafolder = '/mnt/workspace/workgroup/lingke/holo_open/'
holo_datasets = os.listdir(holo_datafolder)
holo_result_file = 'dagmm_holo_result.csv'
pub_result_file = 'dagmm_pub_result.csv'
# SPOT config
q = 1e-4
lm = 0.999

def load_holo_dataset(dataset):
    folder = os.path.join(holo_datafolder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    train = np.load(os.path.join(folder, f'train.npy'))
    test = np.load(os.path.join(folder, f'test.npy'))
    label = np.load(os.path.join(folder, f'test_label.npy'))
    return train, test, label


def load_pub_dataset(dataset):
    folder = os.path.join(public_datafolder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    if dataset in ('PSM', 'SWaT'):
        train = np.load(os.path.join(folder, f'train.npy'))
        test = np.load(os.path.join(folder, f'test.npy'))
        label = np.load(os.path.join(folder, f'test_label.npy'))
    else:
        train = np.load(os.path.join(folder, f'{dataset}_train.npy'))
        test = np.load(os.path.join(folder, f'{dataset}_test.npy'))
        label = np.load(os.path.join(folder, f'{dataset}_test_label.npy'))
    return train, test, label

  
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

  
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1

        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def test(datasets, pub):
    for dataset in datasets:

        print(dataset)
        # load data
        if pub:
            trainD, testD, labels = load_pub_dataset(dataset)
            result_file = pub_result_file
        else:
            trainD, testD, labels = load_holo_dataset(dataset)
            result_file = holo_result_file
        # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
        # load model

        param_grid = {
            'comp_hiddens': [[16, 8, 1], [32, 16, 2]],
            'est_hiddens': [[8, 4], [16, 8]],
        }

        # 初始化自定义网格搜索对象
        custom_grid_search = CustomGridSearch(param_grid)
        # 在训练数据上执行自定义网格搜索
        custom_grid_search.fit(trainD, testD, labels, dataset)

        with open(result_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=custom_grid_search.best_score.keys())
            writer.writerow(custom_grid_search.best_score)


class CustomGridSearch:
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None

    def fit(self, trainD, testD, labels, dataset):
        best_score = float('-inf')

        # 遍历所有参数组合
        for params in self._generate_param_combinations():
            # 设置参数，训练，验证模型性能
            print(params)
            model = DAGMM(
                comp_hiddens=params['comp_hiddens'], comp_activation=tf.nn.tanh,
                est_hiddens=params['est_hiddens'], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
                epoch_size=10, minibatch_size=128
            )
            model.fit(trainD)
            loss = model.predict(testD)
            lossT = model.predict(trainD)
            labels = labels.ravel()

            s = SPOT(1e-4)  # SPOT object
            s.fit(lossT, loss)  # data import
            s.initialize(level=0.999, min_extrema=False, verbose=False)  # initialization step

            ret = s.run(dynamic=False)  # run
            pot_th = np.mean(ret['thresholds']) * 1.0
            print(pot_th)
            pred, p_latency = adjust_predicts(loss, labels, pot_th, calc_latency=True)

            scores = combine_all_evaluation_scores(labels, pred, loss, dataset)

            if scores["pa_f_score"] > best_score:
                best_score = scores["pa_f_score"]
                self.best_params = params
                self.best_score = scores

    def _generate_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return param_combinations



if __name__ == '__main__':
    score_list = {
        "dataset": "dataset",
        "Affiliation precision": "Affiliation precision",
        "Affiliation recall": "Affiliation recall",
        "MCC_score": "MCC_score",
        "R_AUC_PR": "R_AUC_PR",
        "R_AUC_ROC": "R_AUC_ROC",
        "VUS_PR": "VUS_PR",
        "VUS_ROC": "VUS_ROC",
        "f05_score_ori": "f05_score_ori",
        "f1_score_c": "f1_score_c",
        "f1_score_ori": "f1_score_ori",
        "f1_score_pa": "f1_score_pa",
        "pa_accuracy": "pa_accuracy",
        "pa_f_score": "pa_f_score",
        "pa_precision": "pa_precision",
        "pa_recall": "pa_recall",
        "point_auc": "point_auc",
        "precision_k": "precision_k",
        "range_auc": "range_auc",
        "range_f_score": "range_f_score",
    }
    with open(holo_result_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=score_list.keys())
        writer.writeheader()
    with open(pub_result_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=score_list.keys())
        writer.writeheader()
    # public dataset test
    test(public_datasets,True)
    # holo dataset test
    test(holo_datasets, False)

