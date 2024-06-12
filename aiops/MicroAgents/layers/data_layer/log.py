import polars as pl
from .base import Processer
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
import configparser
import json
from functools import partial
from datetime import datetime, timedelta
from .anomaly_detection import ksigma_anomaly_detection
from utils.load import load_model_by_config_name

def log_parsing(log_message,template_miner, new_template_miner):
    result = template_miner.match(log_message)
    if result:
        return result.cluster_id
    else:
        result = new_template_miner.match(log_message)
        if result:
            return -result.cluster_id
        else:
            raise ValueError(f"log parsing failed: {log_message}")
    


class LogExpert:
    def __init__(self, path_to_log_result=None, llm_expert=None, save=False):
        self.cache = {}
        if path_to_log_result:
            with open(path_to_log_result) as f:
                self.cache = json.load(f)
        self.llm_expert = llm_expert
        self.path_to_log_result = path_to_log_result
        self.save_mode = save

    def detect(self, template_id, log_template, log_message):
        if isinstance(template_id, int):
            template_id = str(template_id)
        # print(template_id)
        if template_id in self.cache:
            return self.cache[template_id]
        else:
            res = self.query_llm(template_id, log_message)
            self.cache[template_id] = {'is_abnormal': res, 'template': log_template, 'message_instance': log_message}
            if self.save_mode:
                with open(self.path_to_log_result, 'w') as f:
                    json.dump(self.cache, f)
                print(f'save log expert cache for template {template_id}\n')
            return self.cache[template_id] 

    def query_llm(self, template_id, log_message):
        print(f'Query LLM for template id: {template_id}, template message: {log_message}')
        instructions = ("You are an expert of log analyser. I will provide a log message."
                        "Please tell me whether the log message is abnormal."
                        "Answer with True or False"
                        "Do not use any other words")
        messages = [{
                "role": "system",
                "content": instructions},
            {
                "role": "user",
                "content": f"log message: {log_message}"
            }]
        for i in range(3):
            res, status  = self.llm_expert.reply(messages)
            if status:
                return "true" in res.choices[0].message.content.lower()
        return  False

class LogSummary:
    def __init__(self, llm_expert=None):
        self.llm_expert = llm_expert
    def summary(self, log_message):
        instructions = (
            "You are an expert of log analyser. I will provide some log messages that are probably abnormal."
            "Please analyse and summarize the log messages to better understand the problem." 
            "The summary should be concise but detailed enough to understand the problem."
            "Please output the results in the following format:"
            "A concise summary of the log messages;\n"
            "The state of the module: NORMAL/ABNORMAL;\n"
            "If abnormal, what and why the possible problem is;\n"
        )
        messages = [{
                "role": "system",
                "content": instructions},
            {
                "role": "user",
                "content": f"log message: {log_message}"
            }]
        for i in range(3):
            res, status = self.llm_expert(messages)
            if status:
                return res['choices'][0]['message']['content']
        return  "" 



class Log(Processer):
    def __init__(self, path_to_template, config_file, path_to_log_result, model, **kwargs):
        super().__init__(**kwargs)
        self.path_to_template = path_to_template
        self.config_file = config_file
        config = TemplateMinerConfig()
        config.load(config_file)
        config.profiling_enabled = False
        
        persistence = FilePersistence(path_to_template)
        self.template_miner = TemplateMiner(persistence, config=config)

        path_to_template_new = path_to_template.replace('.bin', '_new.bin')
        persistence_new = FilePersistence(path_to_template_new)
        self.template_miner_new = TemplateMiner(persistence_new, config=config)

        llm_expert = load_model_by_config_name(model)
        self.log_expert = LogExpert(path_to_log_result, llm_expert)
        self.log_summary = LogSummary(llm_expert)
    
    def anomaly_detection(self, recent_df, history_df, time_col='m_timestamp', log_col='m_message', window_size={'minutes':1}, k=3, summary=False, **kwargs):
        
        # recent_df = recent_df.filter(pl.col(log_col).str.lengths()<=1000)
        recent_df = self.parse(recent_df, log_col)
        history_df = self.parse(history_df, log_col)

        counted_recent_df = self.get_counts(recent_df, time_col, log_col, window_size)
        counted_history_df = self.get_counts(history_df, time_col, log_col, window_size={'minutes':2})    

        res = {'new_template':[], 'boom_template':[]}
        for template_id in counted_recent_df['log_template_id'].unique():
            counted_recent_df_template = counted_recent_df.filter(pl.col('log_template_id') == template_id)
            counted_history_df_template = counted_history_df.filter(pl.col('log_template_id') == template_id)
            if counted_history_df_template.shape[0] == 0:

                res['new_template'].append((counted_recent_df_template[log_col][0], template_id))
            else:
                res_template, _ = ksigma_anomaly_detection(counted_recent_df_template, counted_history_df_template, metric_columns=['count_per_window'], k=k)
                if len(res_template['increased_metric']) > 0:
                    # print(counted_recent_df_template)
                    # print(counted_history_df_template)
                    res['boom_template'].append((counted_recent_df_template[log_col][0], template_id))
            
        res_text = ""

        if res['boom_template']:
            res_text += f"There's been an explosion of logs, which may be abnormal: \n"
            for i, l in enumerate(res['boom_template']):
                res_text += f"\t Log {i+1}. {l[0]}\n"

        recent_df = recent_df.filter(pl.col('log_template_id') < 0)
        template_id_index, log_message_idx, raw_log_message_idx = recent_df.columns.index('log_template_id'), recent_df.columns.index(log_col), recent_df.columns.index('raw_m_message')
        recent_df = recent_df.unique(subset=['log_template_id'], keep='last', maintain_order=True)
        content_abnormal_logs = []
        for row in recent_df.iter_rows():
            template_id = row[template_id_index]
            log_message = row[log_message_idx]
            raw_log_message = row[raw_log_message_idx]
            if template_id < 0:
                log_template = self.template_miner_new.drain.id_to_cluster.get(-template_id).get_template()
                llm_res = self.log_expert.detect(template_id, log_template, log_message)
                # print(template_id, llm_res)
                if llm_res['is_abnormal']:
                    content_abnormal_logs.append((log_message, log_template))
        if len(content_abnormal_logs) > 0:
            res_text += f"There are logs with abnormal content: \n"
            for i, log in enumerate(content_abnormal_logs):
                res_text += f"\t Log {i+1}. {log[0]}\n"
        
        if summary and res_text:
            res_text = self.log_summary.summary(res_text)

        abnormal_template = []
        for log, template_id in res['boom_template']:
            if template_id >= 0:
                log_template = self.template_miner.drain.id_to_cluster.get(template_id).get_template()
            elif template_id < 0:
                log_template = self.template_miner_new.drain.id_to_cluster.get(-template_id).get_template()
            abnormal_template.append(log_template)
        for log_message, log_template in content_abnormal_logs:
            abnormal_template.append(log_template)
        
        return res_text, abnormal_template
    
    def parse(self, df, log_col='m_message'):
        df = df.with_columns(pl.col(log_col).apply(partial(log_parsing,template_miner=self.template_miner, new_template_miner=self.template_miner_new), return_dtype=pl.Int64).alias('log_template_id'))
        return df

    
    def get_counts(self, df, time_col='m_timestamp', log_col='m_message', window_size={'minutes':1}):
        
        min_timestamp = df.select(pl.min(time_col)).to_dict(False)[time_col][0]
        window_size = timedelta(**window_size)


        # 添加一个用于分组的列，计算时间戳与最小时间戳的差异，并将这个时间差转换为窗口数量
        df = df.with_columns(
            ((pl.col(time_col) - min_timestamp).dt.seconds().cast(pl.Int64) / 
            window_size.total_seconds()).cast(pl.UInt32).alias("window")
        )

        # 对每个时间窗口和每个template_id进行分组，并计数
        result_df = df.group_by(["window", "log_template_id"]).agg(
            pl.len().alias("count_per_window"),pl.col(log_col).first()
        ).sort(['window','log_template_id'])
        
     
    
        return result_df
    
    


        




