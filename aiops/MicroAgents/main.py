import importlib

import argparse
from loader.nezha import NezhaLoader
import polars as pl
import pickle
from utils import process
from datetime import datetime, timedelta
from agentnn import agentnn
import json
from utils.load import load_data, save_data
import os
from common.prompts.public_english import prompts
import time

get_data_at_time = process.get_data_at_time
AgentNN = agentnn.AgentNN


parser = argparse.ArgumentParser()
parser.add_argument('--organization_mode', type=str, default='convolution')
parser.add_argument('--dataset', type=str, default='GShop')
parser.add_argument('--model', type=str, default='qwen-max')

args = parser.parse_args()



loader = NezhaLoader(filename='./data/nezha')

loader.load_processed_data('./data/nezha/processed_data')

fault_free_time = {
    'GShop': [('2022-08-22 03:50:19', '2022-08-22 03:51:19'), ('2022-08-23 16:59:27', '2022-08-23 17:00:28')],
    'TrainTicket': [('2023-01-29 08:48:16','2023-01-29 08:51:04'), ('2023-01-30 11:38:52', '2023-01-30 11:39:51')],
    'dateformat': '%Y-%m-%d %H:%M:%S',
}

template_files = {
    'GShop': 'hipster.bin',
    'TrainTicket': 'ts.bin'
}
config_files = {
    'GShop': 'drain3_hipster.ini',
    'TrainTicket': 'drain3_ts.ini'
}
log_result_files = {
    'GShop': 'anomaly_hipster.json',
    'TrainTicket': 'anomaly_ts.json'
}
metrics_names = [
 'CpuUsageRate(%)',
 'MemoryUsageRate(%)',
 'SyscallRead',
 'SyscallWrite',
 'NodeMemoryUsageRate(%)',
 'NodeCpuUsageRate(%)',
 'NetworkReceiveBytes',
 'NetworkTransmitBytes',
 'PodClientLatencyP90(s)',
 'PodServerLatencyP90(s)',
 'PodClientLatencyP95(s)',
 'PodServerLatencyP95(s)',
 'PodClientLatencyP99(s)',
 'PodServerLatencyP99(s)'
 ]




for dataset in [args.dataset]:

    df_logs = loader.df.filter(pl.col('system_name') == dataset)
    df_metrics = loader.df_metric_default.filter(pl.col('system_name') == dataset) 
    df_traces = loader.df_trace.filter(pl.col('system_name') == dataset)
    df_labels = loader.df_label.filter(pl.col('system_name') == dataset)
   
    with open(f'./data/nezha/architecture/{dataset}.pkl', 'rb') as f:
        architecture = pickle.load(f)
    
    with open(f'./data/nezha/description.json', 'r') as f:
        system_intro = json.load(f)
        system_intro = system_intro[dataset]['system description']
    
    history_df_logs = loader.df
    history_df_logs = load_data(f'./data/nezha/fault_free_data', f'{dataset}_logs')
    history_df_metrics = load_data(f'./data/nezha/fault_free_data', f'{dataset}_metrics')
    history_df_traces = load_data(f'./data/nezha/fault_free_data', f'{dataset}_traces')
    history_df_metrics = history_df_metrics.fill_nan(0)

    item_nums = len(df_labels)
    print(item_nums)

    agent = AgentNN(system_intro, architecture, False, None, args.model,prompts=prompts,
                    template_file=template_files[dataset], config_file=config_files[dataset], log_result_file=log_result_files[dataset])


    
    for i, row in enumerate(df_labels.iter_rows()):
        
        

        anomaly_time, anomaly_type, anomaly_service = row[7], row[3], row[2]
        print(f'Sample {i}, Time {anomaly_time}, Type {anomaly_type}, Service {anomaly_service}')

        anomaly_time += timedelta(minutes=1)

      
        recent_df_logs = get_data_at_time(df_logs, start_time=anomaly_time, time_window={'minutes':2})
        recent_df_metrics = get_data_at_time(df_metrics, start_time=anomaly_time, time_window={'minutes':2})
        recent_df_traces = get_data_at_time(df_traces, start_time=anomaly_time, time_window={'minutes':2})
        recent_df_metrics = recent_df_metrics.fill_nan(0)

        
        res, description = agent.analyze(history_df_logs, history_df_traces, history_df_metrics, recent_df_logs, recent_df_traces, recent_df_metrics,
                        metric_names=metrics_names, organization_mode=args.organization_mode, time_col="m_timestamp", log_col="m_message", 
                        window_size={'minutes':1}, k=3)
        
        
        res_dir = f'./data/nezha/res/{args.organization_mode}'

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        file_name = f'{dataset}_{i}_{anomaly_service}_{anomaly_type}.txt'
        file = os.path.join(res_dir, file_name)
        with open(file, 'w') as f:
            f.write(res)
            
   