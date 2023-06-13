from .utils.metrics import metricor
from .analysis.robustness_eval import generate_curve
import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def get_range_vus_roc(score, labels, slidingWindow):
    grader = metricor()
    R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, 2*slidingWindow)
    metrics = {'R_AUC_ROC': R_AUC_ROC, 'R_AUC_PR': R_AUC_PR, 'VUS_ROC': VUS_ROC, 'VUS_PR': VUS_PR}

    return metrics
