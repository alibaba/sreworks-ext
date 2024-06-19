    
from datetime import datetime, timedelta
import polars as pl
from .json import parse_json_markdown
import json

def get_data_at_time(df, start_time=None, end_time=None, time_window={'minutes':1}, time_col='m_timestamp', date_format= '%Y-%m-%dT%H:%M:%S.%fZ'):

    if start_time is None and end_time is None:
        raise ValueError("start_time and end_time cannot be both None")
    
    if start_time is not None and isinstance(start_time, str):
        start_time = datetime.strptime(start_time, date_format)
    if end_time is not None and isinstance(end_time, str):
        end_time = datetime.strptime(end_time, date_format)
    
    if time_window is not None and end_time is not None:
        start_time = end_time - timedelta(**time_window)
    if time_window is not None and start_time is not None:
        end_time = start_time + timedelta(**time_window)
        
    return df.filter((pl.col(time_col) >= start_time) & (pl.col(time_col) < end_time))

def get_network_metrics(trace_df, pod_name=None):
    """
    Calculate the 90th percentile of network latency for each PodName in the trace data,
    excluding any 'front' related PodNames.

    :param trace_df: DataFrame containing columns 'TraceID', 'SpanID', 'ParentID', 'PodName', 'EndTimeUnixNano'
    :return: DataFrame containing each PodName and its respective 90th percentile network latency
    """

    if pod_name and 'front' in pod_name:
        return pl.DataFrame({
            "PodName": [pod_name],
            "NetworkP90(ms)": [1.0]
        })
    # Filter out 'front' from PodName first to simplify later computations.
    # non_front_traces = trace_df.filter(~pl.col('PodName').str.contains("front"))

    # Create temporary columns in the dataframe to facilitate a self join on ParentID == SpanID, 
    # excluding matches within the same PodName.
    current = trace_df.rename({
        "ParentID": "parent_span_id", # this is the span it is looking for as parent
        "PodName": "current_pod_name",
        "m_timestamp_end": "current_end_time_stamp"
    })

    if pod_name:
        current = current.filter(pl.col('current_pod_name') == pod_name)

    parent = trace_df.rename({
        "SpanID": "current_span_id", # this is the span ID that might be a parent
        "PodName": "parent_pod_name",
        "m_timestamp_end": "parent_end_time_stamp"
    })

    # Perform the join
    joined_df = current.join(
        parent, left_on=['parent_span_id'], right_on=['current_span_id'], how='inner').filter(~(pl.col('current_pod_name')==pl.col('parent_pod_name')))

    # Calculate latency in microseconds.
    latencies = joined_df.with_columns(
        ((pl.col("parent_end_time_stamp") - pl.col("current_end_time_stamp")) / 1000000).alias("Latency")
    )

    # Group by PodName and calculate the 90th percentile of Latency.
    result_df = latencies.groupby("current_pod_name").agg(
        pl.col("Latency").quantile(0.90).alias("NetworkP90(ms)")
    ).rename({'current_pod_name':'PodName'})

   
    return result_df

def extract_res(res):
    try:
        res = parse_json_markdown(res)
        
    except:
        res = res

def map_module_log(module, dataset, anomaly_type):
    dataset_path = {
        'GShop': './data/nezha/fault_free_data/root_cause_hipster.json',
        'TrainTicket': './data/nezha/fault_free_data/root_cause_ts.json'

    }

    with open(dataset_path[dataset], 'r') as f:
        dataset_dict = json.load(f)

    module = module.rsplit('-',2)[0]
    return dataset_dict[module][anomaly_type]