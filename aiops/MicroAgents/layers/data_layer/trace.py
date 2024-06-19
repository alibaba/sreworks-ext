import polars as pl
from .base import Processer

from .anomaly_detection import ksigma_anomaly_detection, rule_detection, read_mean_and_std

class Trace(Processer):
    def __init__(self):
        super(Trace, self).__init__()
    
    def anomaly_detection(self, recent_df, history_df=None, k=3, mean=None, std=None, metric_columns=['NetworkP90(ms)'], use_rule=False):
        """
        ksigma anomaly detection.
        Args:
            recent_df: recent data
            history_df: history data
            k: ksigma
            mean: mean of history data
            std: std of history data
            metric_columns: columns to detect anomaly
        """
        pod_name = recent_df['PodName'][0]
        mean, std = read_mean_and_std(pod_name)

        res, _ = ksigma_anomaly_detection(recent_df, history_df, metric_columns=metric_columns, k=k, mean=mean, std=std)

        res_text = ""
        if len(res['increased_metric']) > 0:
            res_text = f"The network latency of the module {pod_name} is abnormally high, which indicates that there is possiblely a network delay.\n"
        
        return res_text, res




