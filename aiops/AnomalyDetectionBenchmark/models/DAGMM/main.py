import tensorflow as tf
import numpy as np
import os
import csv
from spot import SPOT

from metrics.combine_all_scores import combine_all_evaluation_scores

from dagmm import DAGMM

public_datasets = ['SMAP', 'MSL', 'SMD', 'NIPS_TS_CCard', 'NIPS_TS_Swan', 'NIPS_TS_Water', 'NIPS_TS_Syn_Mulvar','SWaT']

#datasets folder
public_datafolder = ''
holo_datafolder = ''
holo_datasets = os.listdir(holo_datafolder)
holo_result_file = 'holo_result.csv'
pub_result_file = 'pub_result.csv'
#SPOT config
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

def test(datasets,pub):

    for i in range(len(datasets)):
        print(datasets[i])
        # load data
        if pub:
            trainD, testD, labels = load_pub_dataset(datasets[i])
            result_file = pub_result_file
        else:
            trainD, testD, labels = load_holo_dataset(datasets[i])
            result_file = holo_result_file
        # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
        # load model
        model = DAGMM(
            comp_hiddens=[16,8,1], comp_activation=tf.nn.tanh,
            est_hiddens=[8,4], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
            epoch_size=10, minibatch_size=128
        )

        model.fit(trainD)
        loss = model.predict(testD)
        lossT = model.predict(trainD)
        print(loss.shape,loss[0],max(loss),min(loss))
        labels = labels.ravel()

        # spot method
        s = SPOT(q)  # SPOT object
        s.fit(lossT, loss)  # data import
        s.initialize(level=lm, min_extrema=False, verbose=False)  # initialization step

        ret = s.run(dynamic=False)  # run
        pot_th = np.mean(ret['thresholds']) * 1.0
        print(pot_th)
        pred, p_latency = adjust_predicts(loss, labels, pot_th, calc_latency=True)

        # calculate metrics
        scores = combine_all_evaluation_scores(labels, pred, loss, datasets[i])
        with open(result_file,'a',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=scores.keys())
            writer.writerow(scores)

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
    test(holo_datasets,False)