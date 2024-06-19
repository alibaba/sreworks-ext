import polars as pl
import os
import re


ignore_decreasing_metrics = ['CpuUsageRate(%)','MemoryUsageRate(%)','SyscallRead','SyscallWrite','NetworkP90(ms)']






def read_mean_and_std(pod):
    """
    fun determine_alarm: determin whether violate 3-sgima
    :parameter
        pod - podname to find corrsponding metric threshold file
        metric_type - find correspding column
        metric_vault - compare with the history mean and std
        std_num - constrol std_num * std
    :return
        true - alarm
        false - no alarm
    """
    pod = pod.rsplit('-',2)[0]

    metric_threshold_csv = './data/nezha/metric_threshold/overall.csv'
    hisory_metric = pl.read_csv(metric_threshold_csv)
    mean = hisory_metric.filter(pl.col('PodName') == pod).select([pl.col(col).alias(col.replace("_mean", "")) for col in hisory_metric.columns if col.endswith("_mean")])
    std = hisory_metric.filter(pl.col('PodName') == pod).select([pl.col(col).alias(col.replace("_std", "")) for col in hisory_metric.columns if col.endswith("_std")])

    return mean, std
    

def rule_detection(recent_df, metric_columns, dataset):
    res = {
        'increased_metric': set(),
        'decresed_metric': set(),
        'fluctuate_metric': set()
    }

    for metric_type in metric_columns:
        if metric_type == "CpuUsageRate(%)" or metric_type == 'MemoryUsageRate(%)':
            if recent_df[metric_type].max() > 80:
                res['increased_metric'].add(metric_type)
        else:
            if dataset == "GShop":
                # for hipster
                if recent_df[metric_type].max()  > 200:
                     res['increased_metric'].add(metric_type)
            elif dataset == "TrainTickets":
                # for ts
                if recent_df[metric_type].max() > 300:
                     res['increased_metric'].add(metric_type)
    return res



def ksigma_anomaly_detection(recent_df, history_df=None, k=3, mean=None, std=None, metric_columns=[]):
    """
    ksigma anomaly detection.
    Args:
        recent_df: recent data
        history_df: history data
        k: ksigma
        mean: mean of history data
        std: std of history data
        metric_columns: columns to detect anomaly
        res: anomaly result: {metric_name: anomaly_flag} where anomaly_flag is 1, 0 or -1. 1 means too large, 0 means normal, -1 means too low

    """
    if history_df is None and mean is None and std is None:
            raise ValueError("Either history_df or mean and var must be provided")
    if mean is None or std is None:
        mean = history_df.select(pl.mean(metric_columns))
        std = history_df.select(pl.std(metric_columns))
    

    for column in metric_columns:
        col_mean = mean[column][0]
        col_std = std[column][0]
        if col_std is None:
            col_std = col_mean/2
        try:
        
            upper_bound = col_mean + k * col_std
            lower_bound = col_mean - k * col_std
        except:
            # print(recent_df)
            print(history_df)
            print("error:", column)
        
        

        if upper_bound <= 0 and lower_bound <= 0:
            upper_bound = 500
            lower_bound = 0
        
        if "usage" in column.lower():
            upper_bound = max(upper_bound, 80)
        if column == 'count_per_window':
            upper_bound = max(upper_bound, 10)

        recent_df = recent_df.with_columns(
            pl.when((pl.col(column) > upper_bound))
            .then(1).when((pl.col(column) < lower_bound)).then(-1)
            .otherwise(0)
            .alias(f'{column}_anomaly')
        )
    

    res = {
        'increased_metric': set(),
        'decresed_metric': set(),
        'fluctuate_metric': set()
    }

    

    for col in metric_columns:
        col_mean = mean[col][0]
        col_std = std[col][0]
        if col_std is None:
            col_std = col_mean/2
        # if col_std > col_mean and not ("usage" in col.lower() or 'NetworkP90' in col): #去掉震荡过大的指标
        #     continue

        if recent_df.select(pl.col(f'{col}_anomaly').gt(0).any()).to_series()[0]:
            res['increased_metric'].add(col)
        if recent_df.select(pl.col(f'{col}_anomaly').lt(0).any()).to_series()[0]:
            if col not in ignore_decreasing_metrics:
                res['decresed_metric'].add(col)
            res['decresed_metric'].add(col)
    res['fluctuate_metric'] = res['increased_metric'].intersection(res['decresed_metric'])
    res['increased_metric'] = res['increased_metric'].difference(res['fluctuate_metric'])
    res['decresed_metric'] = res['decresed_metric'].difference(res['fluctuate_metric'])

    return res, recent_df

