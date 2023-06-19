from metrics.f1_score_f1_pa import *
from metrics.fc_score import *
from metrics.precision_at_k import *
from metrics.customizable_f1_score import *
from metrics.AUC import *
from metrics.Matthews_correlation_coefficient import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.models.feature import Window
from metrics.vus.metrics import get_range_vus_roc
import numpy as np

def combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores):
    events_pred = convert_vector_to_events(y_test) 
    events_gt = convert_vector_to_events(pred_labels)
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
    vus_results = get_range_vus_roc(y_test, pred_labels, 100) # default slidingWindow = 100

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
                  "R_AUC_ROC": vus_results["R_AUC_ROC"], 
                  "R_AUC_PR": vus_results["R_AUC_PR"],
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"]
                 }
    
    # score_list_simple = {
    #               "pa_accuracy":pa_accuracy, 
    #               "pa_precision":pa_precision, 
    #               "pa_recall":pa_recall, 
    #               "pa_f_score":pa_f_score,
    #               "MCC_score":MCC_score, 
    #               "Affiliation precision": affiliation['precision'], 
    #               "Affiliation recall": affiliation['recall'],
    #               "R_AUC_ROC": vus_results["R_AUC_ROC"], 
    #               "R_AUC_PR": vus_results["R_AUC_PR"],
    #               "VUS_ROC": vus_results["VUS_ROC"],
    #               "VUS_PR": vus_results["VUS_PR"]
    #               }
    
    # return score_list, score_list_simple
    # return score_list_simple
    return score_list


if __name__ == '__main__':
    y_test = np.load("data/events_pred_MSL.npy")+0
    pred_labels = np.load("data/events_gt_MSL.npy")+0
    anomaly_scores = np.load("data/events_scores_MSL.npy")
    print(len(y_test), max(anomaly_scores), min(anomaly_scores))
    score_list_simple = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)

    for key, value in score_list_simple.items():
        print('{0:21} :{1:10f}'.format(key, value))