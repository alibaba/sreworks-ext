import os

import numpy as np
import pandas as pd
from f1_score_f1_pa import *
from fc_score import *
from precision_at_k import *
from customizable_f1_score import *
from AUC import *
from Matthews_correlation_coefficient import *
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
# from vus.models.feature import Window
from vus.metrics import get_range_vus_roc
from sklearn.metrics import f1_score



def combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores):
    events_pred = convert_vector_to_events(y_test) # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(pred_labels)     # [(3, 4), (7, 10)]
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    true_events = get_events(y_test)
    _, _, _, f1_score_ori, f05_score_ori = get_accuracy_precision_recall_fscore(y_test, pred_labels)
    f1_score_pa = get_point_adjust_scores(y_test, pred_labels, true_events)[5]
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(y_test, pred_labels)
    range_f_score = customizable_f1_score(y_test, pred_labels)
    _, _, f1_score_c = get_composite_fscore_raw(y_test, pred_labels,  true_events, return_prec_rec=True)
    precision_k = precision_at_k(y_test, anomaly_scores, pred_labels)
    point_auc = point_wise_AUC(pred_labels, y_test)
    range_auc = Range_AUC(pred_labels, y_test)
    MCC_score = MCC(y_test, pred_labels)
    results = get_range_vus_roc(y_test, pred_labels, 100) # slidingWindow = 100 default

    
    score_list = {"f1_score_ori": f1_score_ori, 
                  "f05_score_ori" : f05_score_ori, 
                  "f1_score_pa": f1_score_pa,
                  "pa_accuracy":pa_accuracy, 
                  "pa_precision":pa_precision, 
                  "pa_recall":pa_recall, 
                  "pa_f_score":pa_f_score,
                  "range_f_score": range_f_score,
                  "f1_score_c": f1_score_c, 
                  "precision_k": precision_k,
                  "point_auc": point_auc,
                  "range_auc": range_auc, 
                  "MCC_score":MCC_score, 
                  "Affiliation precision": affiliation['precision'], 
                  "Affiliation recall": affiliation['recall'],
                  "R_AUC_ROC": results["R_AUC_ROC"], 
                  "R_AUC_PR": results["R_AUC_PR"],
                  "VUS_ROC": results["VUS_ROC"], 
                  "VUS_PR": results["VUS_PR"]}
    
    return score_list


# def main():
#     y_test = np.zeros(100)
#     y_test[10:20] = 1
#     y_test[50:60] = 1
#     pred_labels = np.zeros(100)
#     pred_labels[15:17] = 1
#     pred_labels[55:62] = 1
#     anomaly_scores = np.zeros(100)
#     anomaly_scores[15:17] = 0.7
#     anomaly_scores[55:62] = 0.6
#     pred_labels[51:55] = 1
#     true_events = get_events(y_test)
#     scores = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)
#     # scores = test(y_test, pred_labels)
#     for key,value in scores.items():
#         print(key,' : ',value)

def threshold(score_test, y_test):
    thres_grid = np.linspace(start=np.min(score_test), stop=np.max(score_test), num=100)
    f1_grid = []
    for thres in thres_grid:
        pred_test = np.zeros_like(y_test)
        pred_test[score_test > thres] = 1
        f1 = f1_score(y_true=y_test, y_pred=pred_test)
        f1_grid.append(f1)

    



def main():
    dataset_list = os.listdir('../public_datasets')
    model_list = ['KNN', 'LOF', 'IForest','COPOD', 'ECOD',
                  'DeepSVDD', 'LSTM', 'LSTM_AE', 'LSTM_VAE']

    df_metric = pd.DataFrame(data=None, index=dataset_list, columns=model_list)

    for dataset in dataset_list:
        for model in model_list:
            result = np.load('../result/' + dataset + '_' + model + '.npz', allow_pickle=True)
            y_test = result['y_test']
            score_test = result['score_test']

            break

    scores = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)
    for key,value in scores.items():
        print(key,' : ',value)

    
if __name__ == "__main__":
    main()