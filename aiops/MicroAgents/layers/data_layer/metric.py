import polars as pl
from .base import Processer
from .anomaly_detection import ksigma_anomaly_detection, read_mean_and_std, rule_detection

class Metric(Processer):
    def __init__(self, **kwargs):
        super(Metric, self).__init__(**kwargs)

    def anomaly_detection(self, recent_df, history_df=None, metric_columns=[], k=3, mean=None, std=None, use_rule=False):
        pod_name = recent_df['PodName'][0]
        dataset = recent_df['system_name'][0]
        mean, std = read_mean_and_std(pod_name)
        
        if use_rule:
            res = rule_detection(recent_df, metric_columns, dataset)
        else:
            res, _ = ksigma_anomaly_detection(recent_df, history_df, metric_columns=metric_columns, k=k, mean=mean, std=std)
        res_text = ""
        
        if len(res['increased_metric']) > 0:
            res_text += f"There are some metrics abnormally larger than usual: "
            for metric in res['increased_metric']:
                res_text += f"{metric},"
            res_text += "\n"
       

        return res_text, res['increased_metric']



        
        


    

        


