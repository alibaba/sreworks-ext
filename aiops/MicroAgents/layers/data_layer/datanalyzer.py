from .log import Log
from .trace import Trace
from .metric import Metric
import os
import json

root_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

with open(os.path.join(root_path, 'common/configs/global.json'), 'r') as f:
    global_config = json.load(f)

class DataAnalyzer:
    def __init__(self, data_type=["log", "trace", "metric"], template_file="", config_file="", log_result_file="", model=""):
        self.data_type = data_type


        for data in data_type:
            if data == "log":
                path_to_template = os.path.join(root_path, global_config['log_template_config_path'],template_file)
                path_to_log_result = os.path.join(root_path,  global_config['log_template_config_path'], log_result_file)
                config_file = os.path.join(root_path,  global_config['log_template_config_path'], config_file)
                self.log = Log(path_to_template, config_file, path_to_log_result, model)
            elif data == "trace":
                self.trace = Trace()
            elif data == "metric":
                self.metric = Metric()
    
    def analyze(self, logs, traces, metrics, use_rule=False):

        res_text = ""
        symptoms = []

        for data in self.data_type:
            if data == "log":
                history_df, recent_df, time_col, log_col, window_size, k, summary = logs.get('history_df'), logs.get('recent_df'), logs.get('time_col'), logs.get('log_col'), logs.get('window_size'), logs.get('k'), logs.get('summary')
                if recent_df.shape[0] == 0 or history_df.shape[0] == 0:
                    continue
                res, anomaly_logs = self.log.anomaly_detection(recent_df, history_df, time_col=time_col, log_col=log_col, window_size=window_size, k=k, summary=summary)
                if res:
                    res_text += f"LOG OBSERVATION:The log monitor detect abnormal log behavior:\n{res}"
                    symptoms.extend(anomaly_logs)

            elif data == "trace":
                history_df, recent_df, k, metric_columns = traces.get('history_df'), traces.get('recent_df'), traces.get('k'), traces.get('metric_columns')
                if recent_df.shape[0] == 0 or history_df.shape[0] == 0:
                    continue
                res, _ = self.trace.anomaly_detection(recent_df, history_df=history_df, k=k, mean=None, std=None, metric_columns=metric_columns)
                if res:
                    res_text += f"TRACE OBSERVATION: The trace monitor detect abnormal trace behavior:\n{res}"
                    symptoms.extend('NetworkP90(ms)')
            elif data == "metric":
                history_df, recent_df, k, metric_columns = metrics.get('history_df'), metrics.get('recent_df'), metrics.get('k'), metrics.get('metric_columns')
                if recent_df.shape[0] == 0 or history_df.shape[0] == 0:
                    continue
                res, anomaly_metrics = self.metric.anomaly_detection(recent_df, history_df=history_df, metric_columns=metric_columns, k=k, mean=None, std=None, use_rule=use_rule)
                if res:
                    res_text += f"METRIC OBSERVATION: The metric monitor detect abnormal metric behavior:\n{res}"
                    symptoms.extend(anomaly_metrics)
        return res_text, symptoms