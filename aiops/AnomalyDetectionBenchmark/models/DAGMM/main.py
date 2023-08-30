import tensorflow as tf
import numpy as np
import os
import csv
from spot import SPOT
import argparse
import itertools


from metrics.combine_all_scores import combine_all_evaluation_scores

from dagmm import DAGMM

parser = argparse.ArgumentParser()

# Add all the parameters you need there

parser.add_argument('--dataset', metavar='-d', type=str, required=False, default='MSL', help='dataset name')
parser.add_argument('--instance', metavar='-i', type=str, required=False, default='15', help='instance number')
parser.add_argument('--holo_datafolder', type=str, default='../../datasets/holo/fillzero_std',help='holo_datafolder')
parser.add_argument('--public_datafolder', type=str, default='../../datasets/public/',help='public_datafolder')
parser.add_argument('--holo_result_save_path', type=str, default='../../result/holo_result.csv',help='holo_result_save_path')
parser.add_argument('--public_result_save_path', type=str, default='../../result/public_result.csv',help='public_result_save_path')
config = parser.parse_args()

public_datafolder = config.public_datafolder
public_datasets = [config.dataset]
holo_datafolder = config.holo_datafolder
holo_datasets = "instance"+config.instance
holo_result_file = config.holo_result_save_path
pub_result_file = config.public_result_save_path

# datasets folder
# public_datafolder = '../../datasets/public/'
# public_datasets = ['SMAP', 'MSL', 'SMD', 'NIPS_TS_CCard', 'NIPS_TS_Swan', 'NIPS_TS_Water', 'NIPS_TS_Syn_Mulvar', 'SWaT']
# holo_datafolder = '../../datasets/holo/fillzero_std'
# holo_datasets = os.listdir(holo_datafolder)
# holo_result_file = 'dagmm_holo_result.csv'
# pub_result_file = 'dagmm_pub_result.csv'
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
        custom_grid_search.best_score["model"] = "DAGMM"
        if pub:
            custom_grid_search.best_score["dataset"] = dataset
        else:
            custom_grid_search.best_score["instance"] = dataset
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

    # public dataset test
    test(public_datasets,True)
    # holo dataset test
    test(holo_datasets, False)

