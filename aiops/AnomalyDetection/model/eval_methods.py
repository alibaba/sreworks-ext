# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
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


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t

def searchThreshold(score,label):
    mins=np.min(score)
    maxs=np.max(score)
    stride=(maxs-mins)/1000
    threshold=-1
    maxF1=[0]*7
    i=mins+stride
    precisions=[]
    recalls=[]
    while i<maxs+stride:
        pred, p_latency = adjust_predicts(score, label, i, calc_latency=True)
        p_t = calc_point2point(pred, label)
        precisions.append(p_t[1])
        recalls.append(p_t[2])
        if p_t[0]>maxF1[0]:
            maxF1=p_t
            threshold=i
        i+=stride
    score=np.array(score)
    mask=score<threshold
    num=mask.sum()
    ratio=num/score.size
    AUPR=0.
    sumv=[]
    for i in range(1,len(recalls)):
        AUPR+=(recalls[i]-recalls[i-1])*precisions[i-1]
        sumv.append((recalls[i]-recalls[i-1])*precisions[i-1])
    return {
        'pot-f1': maxF1[0],
        'pot-precision': maxF1[1],
        'pot-recall': maxF1[2],
        'pot-TP': maxF1[3],
        'pot-TN': maxF1[4],
        'pot-FP': maxF1[5],
        'pot-FN': maxF1[6],
        'pot-threshold': threshold,
        'threshold-ratio':ratio,
        "AUPR":AUPR
    }

def getF1score(score,label,threshold):
    pred, p_latency = adjust_predicts(score, label, threshold, calc_latency=True)
    p_t = calc_point2point(pred, label)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': threshold,
        'pot-latency': p_latency
    }