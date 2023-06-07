##
# 
#Copyright (c) 2023, Alibaba Group;
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#

import re
import time
import pandas as pd
import os
from itertools import combinations
import math
import random
from utils.preprocess import *
from torch.utils.data import Dataset
from collections import OrderedDict

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'description': 'Hadoop distributed file system log',
        'distance_threshold': 0.005,
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+'],
        'description': 'Hadoop mapreduce job log',
        'distance_threshold': 0.08,
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'description': 'Spark job log',
        'distance_threshold': 0.05,
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'description': 'ZooKeeper service log',
        'distance_threshold': 0.03,
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'description': 'Blue Gene/L supercomputer log',
        'distance_threshold': 0.07,
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'description': 'High performance cluster log',
        'distance_threshold': 0.06,
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'description': 'Thunderbird supercomputer log',
        'distance_threshold': 0.18,
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'description': 'Windows event log',
        'distance_threshold': 0.04,
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'description': 'Linux system log',
        'distance_threshold': 0.14,
        },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'description': 'Android framework log',
        'distance_threshold': 0.02,
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'description': 'Health app log',
        'distance_threshold': 0.04,
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'description': 'Apache web server error log',
        'distance_threshold': 0.01,
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'description': 'Proxifier software log',
        'distance_threshold': 0.04,
        },
        
    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'description': 'OpenSSH server log',
        'distance_threshold': 0.002,
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'description': 'OpenStack infrastructure log',
        'distance_threshold': 0.02,
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'description': 'Mac OS log',
        'distance_threshold': 0.05,
        }
}

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    # print(splitters)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers, logformat, max_len=10000000):
    """ Function to transform log file to dataframe 
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r', errors='ignore') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
                if linecount>=max_len:
                    print("The number of lines of logs exceeds ",max_len)
                    break
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def load_train_log(test_log_type=None,benchmark_settings={}):
    """
        加载日志数据
        return 
    """
    print("正在加载日志数据...")
    start_time = int(time.time())

    df_log = None

    for log_type in benchmark_settings:
        if log_type != test_log_type:
            log_format = benchmark_settings[log_type]['log_format']

            # 加载dataframe类型的日志数据
            headers, regex = generate_logformat_regex(log_format)
            df_data = log_to_dataframe(os.path.join("./logs/"+log_type+"/", log_type+"_2k.log"), regex, headers, log_format)
            
            log_rex = benchmark_settings[log_type]['regex']
            df_data['Content'] = df_data['Content'].apply(lambda x : add_var_token(log_rex,x))
            
            if df_log is None:
                df_log = df_data['Content']
            else:
                # df_log.append(df_data['Content'])
                df_log = pd.concat([df_log,df_data['Content']], ignore_index=True)

        # 生成日志语料库的list
        
        # for idx, line in df_log.iterrows():
        #     # log_temp = line['Level']+' '+line['Component'] +': '+line['Content']
        #     log_temp = line['Content']
        #     # corpus.append(log_temp.lower())
        #     corpus.append(log_temp)
    end_time = int(time.time())
    print("加载日志数据耗时%s秒" % (end_time-start_time))

    return df_log

def load_test_log(log_type,benchmark_settings):
    """log_type is choice of: Andriod Apache BGL HDFS HPC Hadoop HealthApp Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper
        加载日志数据
        return 
    """
    print("正在加载日志数据...")
    start_time = int(time.time())
    # if log_type=="HDFS":
    #     log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    # elif log_type=="Windows":
    #     log_format = '<Date> <Time> <Level> <Component> <Content>'
    # elif log_type=="HPC":
    #     log_format = '<Logld> <Node> <Component> <State> <Time> <Flag> <Content>'

    log_format = benchmark_settings[log_type]['log_format']

    # 加载dataframe类型的日志数据
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(os.path.join("./logs/"+log_type+"/", log_type+"_2k.log"), regex, headers, log_format)
    
    log_rex = benchmark_settings[log_type]['regex']
    df_log['Content'] = df_log['Content'].apply(lambda x : (add_var_token(log_rex,x)))

    # 生成日志语料库的list
    corpus = []
    for idx, line in df_log.iterrows():
        # log_temp = line['Level']+' '+line['Component'] +': '+line['Content']
        temp_log = line['Content']
        # corpus.append(log_temp.lower())

        corpus.append(temp_log)
    end_time = int(time.time())
    print("加载日志数据耗时%s秒" % (end_time-start_time))

    # log_rex = benchmark_settings[log_type]['regex']
    # corpus = [add_var_token(log_rex,s) for s in corpus]

    return df_log, corpus


def generate_positive_samples(test_log_type=None, benchmark_settings=None):
    """
        生成postive pair，每个positive piar为同一模版下的日志对
    """
    print("正在生成positive corpus...")
    start_time = int(time.time())
    positive_samples = {}

    for log_type in benchmark_settings:
        if log_type != test_log_type and log_type!='Flink' and log_type!='ODPS':
            # df_log_structured = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_structured.csv")
            # df_log_template = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_templates.csv")
            
            df_log_structured = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_structured.csv")
            df_log_template = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_templates.csv")

            df_log_template = df_log_template.drop_duplicates(subset=['EventId'])

            samples = {}

            for idx, line in df_log_template.iterrows():
                    temp_id = line['EventId']

                    temp_log = df_log_structured[df_log_structured['EventId']==temp_id]
                    temp_log = temp_log['Content'].to_list()

                    log_rex = benchmark_settings[log_type]['regex']
                    temp_log = [(add_var_token(log_rex,s)) for s in temp_log]

                    if len(temp_log)>=2:
                        for pairs in combinations(temp_log,2):
                            if pairs[0]!=pairs[1]:
                                if not temp_id in samples.keys():
                                    samples[temp_id]=OrderedDict()
                                
                                # pairs = tuple(pairs)
                                reverse_pairs = tuple([pairs[1],pairs[0]])

                                if not (pairs in samples[temp_id]) and not (reverse_pairs in samples[temp_id]):
                                    samples[temp_id][pairs] = 0

                    if temp_id in samples.keys():
                        samples[temp_id] = list(samples[temp_id].keys())

            positive_samples[log_type] = samples
            
    if test_log_type=='Flink' or test_log_type=='ODPS':
        
        log_type = 'Flink' if test_log_type=='ODPS' else 'ODPS'
        print("Loading",log_type,"positive pairs...")
        
        log_path = './'+log_type.lower()+'_test.csv'
        df_log = pd.read_csv(log_path)
        df_labeled = df_log[df_log['label_id']!=-1]
        
        positive_event = list(df_labeled['label_id'].value_counts().index)
    
        samples = {}

        for temp_id in positive_event:

            df_temp = df_labeled[df_labeled['label_id']==temp_id]
            df_temp = df_temp.sample(frac=1.0, random_state=42)

            temp_log = df_temp['Content'].to_list()

            if len(temp_log)>=2:
                for pairs in combinations(temp_log,2):
                    if pairs[0]!=pairs[1]:
                        if not temp_id in samples.keys():
                            samples[temp_id]=OrderedDict()
                        
                        reverse_pairs = tuple([pairs[1],pairs[0]])

                        if not (pairs in samples[temp_id]) and not (reverse_pairs in samples[temp_id]):
                            samples[temp_id][pairs] = 0

            if temp_id in samples.keys():
                samples[temp_id] = list(samples[temp_id].keys())
        
        positive_samples[log_type] = samples

    positive_corpus = []
    # 关键
    all_event = OrderedDict()

    for d in positive_samples:
        # print(d)
        for e in positive_samples[d]:
            all_event[(d,e)] = len(positive_samples[d][e])
            for pairs in positive_samples[d][e]:
            # print(i)
                positive_corpus.append(pairs)
    
    end_time = int(time.time())
    print("生成positive corpus耗时%s秒" % (end_time-start_time))

    return positive_corpus, all_event, positive_samples


def generate_neutral_samples(test_log_type=None, positive_corpus=[], benchmark_settings={}):
    """
        生成neutral pair，每个neutral pair为同一个系统，不同模版下的日志对
    """
    print("正在生成neutral corpus...")
    start_time = int(time.time())
    # dataset_corpus = []
    neutral_corpus = set()

    # positive_corpus = set(positive_corpus)

    neutral_nums = 150000
    sub_nums = int(neutral_nums/16) if test_log_type is None else int(neutral_nums/15)
    
    for log_type in benchmark_settings:
        if log_type != test_log_type:
            # print(log_type)
            df_log_structured = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_structured.csv")
            df_log_template = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_templates.csv")

            df_log_template = df_log_template.drop_duplicates(subset=['EventId'])

            samples = {}

            dataset_event = []

            for idx, line in df_log_template.iterrows():
                dataset_event.append(line['EventId'])

            subsub_nums = int(sub_nums/math.comb(len(dataset_event),2))

            for event_pairs in combinations(dataset_event,2):
                pair1_corpus = []
                pair2_corpus = []

                for idx, line in df_log_structured[df_log_structured['EventId']==event_pairs[0]].iterrows():
                    temp_log = line['Content']
                    pair1_corpus.append(temp_log)
                    

                for idx, line in df_log_structured[df_log_structured['EventId']==event_pairs[1]].iterrows():
                    temp_log = line['Content']
                    pair2_corpus.append(temp_log)
                    
                log_rex = benchmark_settings[log_type]['regex']
                pair1_corpus = [add_var_token(log_rex,s) for s in pair1_corpus]
                pair2_corpus = [add_var_token(log_rex,s) for s in pair2_corpus]

                random.shuffle(pair1_corpus)
                random.shuffle(pair2_corpus)

                count = 0

                for i in range(len(pair1_corpus)):
                    for j in range(len(pair2_corpus)):
                        pairs = [pair1_corpus[i],pair2_corpus[j]]
                        reverse_pairs = pairs[::-1]

                        pairs = tuple(pairs)
                        reverse_pairs = tuple(reverse_pairs)

                        if pairs[0]!=pairs[1] and not (pairs in neutral_corpus) and not (reverse_pairs in neutral_corpus):
                            neutral_corpus.add(pairs)
                            count += 1

                        if count>=subsub_nums:
                            break
                    if count>=subsub_nums:
                            break

    end_time = int(time.time())
    print("生成neutral corpus耗时%s秒" % (end_time-start_time))

    return list(neutral_corpus)

def generate_negetive_samples(test_log_type=None, positive_corpus=[], neutral_corpus=[], benchmark_settings={}):
    """
        生成negetive pair，每个negetuve pair为不同系统下的日志对
    """
    print("正在生成negetive corpus...")
    start_time = int(time.time())
    # df_log = None
    all_dataset = []

    # all_corpus = []

    # positive_corpus = set(positive_corpus)
    # neutral_corpus = set(neutral_corpus)

    for log_type in benchmark_settings:
        if log_type != test_log_type:
            all_dataset.append(log_type)

    random.seed(42)

    negetive_corpus = set()

    count = 0
    negetive_nums = 160000
    sub_nums = int(negetive_nums/math.comb(len(all_dataset),2))
    index = 0

    for dataset_pairs in combinations(all_dataset,2):
        pair1_corpus = []
        pair2_corpus = []

        df_log_structured = pd.read_csv("./logs/"+dataset_pairs[0]+"/"+dataset_pairs[0]+"_2k.log_structured.csv")

        for idx, line in df_log_structured.iterrows():
            temp_log = line['Content']
            pair1_corpus.append(temp_log)
            
        log_rex = benchmark_settings[dataset_pairs[0]]['regex']
        pair1_corpus = [add_var_token(log_rex,s) for s in pair1_corpus]

        df_log_structured = pd.read_csv("./logs/"+dataset_pairs[1]+"/"+dataset_pairs[1]+"_2k.log_structured.csv")

        for idx, line in df_log_structured.iterrows():
            temp_log = line['Content']
            pair2_corpus.append(temp_log)
        
        log_rex = benchmark_settings[dataset_pairs[1]]['regex']
        pair2_corpus = [add_var_token(log_rex,s) for s in pair2_corpus]

        random.shuffle(pair1_corpus)
        random.shuffle(pair2_corpus)

        count = 0

        while count<sub_nums:
            
            pairs = [pair1_corpus[index],pair2_corpus[index]]
            reverse_pairs = pairs[::-1]

            pairs = tuple(pairs)
            reverse_pairs = tuple(reverse_pairs)

            if pairs[0]!=pairs[1] and not (pairs in negetive_corpus) and not (reverse_pairs in negetive_corpus):
                negetive_corpus.add(pairs)
                count += 1

            index += 1

            if index>=len(pair1_corpus) or index>=len(pair2_corpus):
                index = 0
                random.shuffle(pair1_corpus)
                random.shuffle(pair2_corpus)

    end_time = int(time.time())
    print("生成negetive corpus耗时%s秒" % (end_time-start_time))
    return list(negetive_corpus)

def generate_contrastive_samples(positive_samples, all_event, batch_size, max_len=100000):
    """
        生成对比学习的正负例样本，正例由postive pair构建，负例由batch内的其他log构建。保证一个batch内同一模版下的positive pair只出现一次
    """
    print("正在生成contrastive corpus...")
    start_time = int(time.time())

    remain_event = all_event

    contrastive_corpus = []

    max_len = max_len
    
    # random.seed(42)

    while len(remain_event)>=batch_size:
        event_list = list(remain_event.keys())
        random.shuffle(event_list)
        # print(event_list)
        for i in range(len(event_list)//batch_size):
            event_pairs = event_list[i*batch_size:i*batch_size+batch_size]
            for event in event_pairs:
                positive_pair = positive_samples[event[0]][event[1]][0]
                positive_samples[event[0]][event[1]] = positive_samples[event[0]][event[1]][1:]
                contrastive_corpus.append(positive_pair)
                remain_event[event] -= 1
                if remain_event[event]==0:
                    del remain_event[event]
                    
            if len(contrastive_corpus)>=max_len:
                break
        if len(contrastive_corpus)>=max_len:
            break


    end_time = int(time.time())
    print("生成contrastive corpus耗时%s秒" % (end_time-start_time))

    return contrastive_corpus

def generate_contrastive_samples_new(positive_samples, all_event, batch_size, max_len=100000):
    """
    """
    print("正在生成contrastive corpus...")
    start_time = int(time.time())

    contrastive_corpus = []

    max_len = max_len
    
    event_list = list(all_event.keys())
    # 按顺序循环取positive_pair,记录index
    positive_index = dict.fromkeys(event_list,0)
    
    random.seed(42)
    
    while len(contrastive_corpus)<=max_len:
        random.shuffle(event_list)
        for i in range(len(event_list)//batch_size):
            events_in_batch = event_list[i*batch_size:i*batch_size+batch_size]
            for event in events_in_batch:
                positive_pair = positive_samples[event[0]][event[1]][positive_index[event]]
                positive_index[event] += 1
                if positive_index[event] >= len(positive_samples[event[0]][event[1]]):
                    positive_index[event] = 0
                contrastive_corpus.append(positive_pair)
                
            if len(contrastive_corpus)>=max_len:
                break

    end_time = int(time.time())
    print("生成contrastive corpus耗时%s秒" % (end_time-start_time))

    return contrastive_corpus

def industry_positive_samples(log_path,batch_size):
    """生成工业数据集的日志对
    """
    print("正在生成positive corpus...")
    start_time = int(time.time())

    df_log = pd.read_csv(log_path)
    df_labeled = df_log[df_log['label_id']!=-1]

    log_select_num = 4

    positive_event = list(df_labeled['label_id'].value_counts().index)

    positive_samples = {}
    samples = {}

    for temp_id in positive_event:

        # if len(samples)>=batch_size:
        #     break

        df_temp = df_labeled[df_labeled['label_id']==temp_id]
        df_temp = df_temp.sample(frac=1.0, random_state=42)

        temp_log = df_temp['Content'].iloc[:log_select_num].to_list()
        # temp_log = df_temp['Content'].to_list()

        if len(temp_log)>=2:
            for pairs in combinations(temp_log,2):
                if pairs[0]!=pairs[1]:
                    if not temp_id in samples.keys():
                        samples[temp_id]=set()
                    
                    # pairs = tuple(pairs)
                    reverse_pairs = tuple([pairs[1],pairs[0]])

                    # if not (pairs in samples[temp_id]) and not (reverse_pairs in samples[temp_id]):
                    #     samples[temp_id].add(pairs)
                    samples[temp_id].add(pairs)

        if temp_id in samples.keys():
            samples[temp_id] = list(samples[temp_id])

    if len(samples)<batch_size:
        print("Positive samples len:",len(samples))
        print("Cannot generate enough positive samples!")
        raise

    positive_samples['industry'] = samples

    positive_corpus = []
    all_event = {}

    for d in positive_samples:
        # print(d)
        for e in positive_samples[d]:
            all_event[(d,e)] = len(positive_samples[d][e])
            for pairs in positive_samples[d][e]:
            # print(i)
                positive_corpus.append(pairs)

    end_time = int(time.time())
    print("生成positive corpus耗时%s秒" % (end_time-start_time))

    return positive_corpus, all_event, positive_samples

def load_event_log(test_log_type=None, benchmark_settings=None, model=None):
    """
        加载日志数据，并生成log token->event id的映射，用于后续查找log对应的center
    """

    all_event_log = {}
    log_to_event = {}

    for log_type in benchmark_settings:
        if log_type != test_log_type and log_type!='Flink' and log_type!='ODPS':
        # if log_type == 'HPC':
            df_log_structured = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_structured.csv")
            df_log_template = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_templates.csv")

            df_log_template = df_log_template.drop_duplicates(subset=['EventId'])

            for idx, line in df_log_template.iterrows():
                    temp_id = line['EventId']
                    

                    temp_log = df_log_structured[df_log_structured['EventId']==temp_id]
                    temp_log = temp_log['Content'].to_list()

                    # temp_log = [add_blank_token(s) for s in temp_log]
                    log_rex = benchmark_settings[log_type]['regex']
                    temp_log = [(add_var_token(log_rex,s)) for s in temp_log]
                    # temp_log = [(add_var_token(log_rex,s)) for s in temp_log]

                    event_id = log_type+temp_id

                    all_event_log[event_id] = temp_log

                    log_tokens = model.tokenize(temp_log)
                    
                    for i in range(len(log_tokens['input_ids'])):
                        log_token = log_tokens['input_ids'][i].cpu().numpy()
                        token_mask = log_tokens['attention_mask'][i].cpu().numpy()
                        log_token = log_token[token_mask!=0]
                        log_to_event[tuple(log_token.tolist())] = event_id

                    # for log_token in log_tokens['input_ids']:
                    #     log_token = log_token.cpu().numpy()
                    #     log_token = log_token[log_token!=0]
                    #     log_to_event[tuple(log_token.tolist())] = event_id

    if test_log_type=='Flink' or test_log_type=='ODPS':
    
        log_type = 'Flink' if test_log_type=='ODPS' else 'ODPS'
        print("Loading",log_type,"event...")
        
        log_path = './'+log_type.lower()+'_test.csv'
        df_log = pd.read_csv(log_path)
        df_labeled = df_log[df_log['label_id']!=-1]
        
        label_ids = df_labeled['label_id'].unique()
        
        for temp_id in label_ids:
            temp_log = df_labeled[df_labeled['label_id']==temp_id]
            temp_log = temp_log['Content'].to_list()
            
            all_event_log[temp_id] = temp_log
            
            log_tokens = model.tokenize(temp_log)
                        
            for i in range(len(log_tokens['input_ids'])):
                log_token = log_tokens['input_ids'][i].cpu().numpy()
                token_mask = log_tokens['attention_mask'][i].cpu().numpy()
                log_token = log_token[token_mask!=0]
                log_to_event[tuple(log_token.tolist())] = temp_id
        
                        
    return all_event_log, log_to_event

def load_event_log_industrial(log_path,model=None):
    
    all_event_log = {}
    log_to_event = {}
    
    df_log = pd.read_csv(log_path)
    df_labeled = df_log[df_log['label_id']!=-1]
    
    label_ids = df_labeled['label_id'].unique()
    
    # print(label_ids)
    
    for temp_id in label_ids:
        temp_log = df_labeled[df_labeled['label_id']==temp_id]
        temp_log = temp_log['Content'].to_list()
        
        all_event_log[temp_id] = temp_log
        
        log_tokens = model.tokenize(temp_log)
                    
        for i in range(len(log_tokens['input_ids'])):
            log_token = log_tokens['input_ids'][i].cpu().numpy()
            token_mask = log_tokens['attention_mask'][i].cpu().numpy()
            log_token = log_token[token_mask!=0]
            log_to_event[tuple(log_token.tolist())] = temp_id
    
    return all_event_log, log_to_event

def load_hadoop_log(log_file,benchmark_settings):
    """
        读取原始的Hadoop日志文件，划分为训练集和测试集，用于异常检测
    """
    log_class = {
        'WordCount':{
            'Normal':['application_1445087491445_0005','application_1445087491445_0007','application_1445175094696_0005'],
            'Machine down':['application_1445087491445_0001','application_1445087491445_0002','application_1445087491445_0003'],
            'Network disconnection':['application_1445175094696_0001','application_1445175094696_0002','application_1445175094696_0003'],
            'Disk full':['application_1445182159119_0001','application_1445182159119_0002','application_1445182159119_0003']
        },
        'PageRank':{
            'Normal':['application_1445062781478_0011','application_1445062781478_0016','application_1445062781478_0019'],
            'Machine down':['application_1445062781478_0012','application_1445062781478_0013','application_1445062781478_0014'],
            'Network disconnection':['application_1445144423722_0020','application_1445144423722_0022','application_1445144423722_0023'],
            'Disk full':['application_1445182159119_0011','application_1445182159119_0013','application_1445182159119_0014']
        }
    }

    log_format = benchmark_settings['Hadoop']['log_format']

    headers, regex = generate_logformat_regex(log_format)

    df_log = None

    for app in log_class:
        for status in log_class[app]:
            for file_name in log_class[app][status]:
                # print(os.listdir(log_file+'/'+file_name))
                for log_name in [i for i in os.listdir(log_file+'/'+file_name) if not i.startswith("._")]:
                    # print(os.path.join(log_file+'/'+file_name+'/'+log_name))
                    df_data = log_to_dataframe(os.path.join(log_file,file_name,log_name), regex, headers, log_format)
                    if status=='Normal':
                        df_data['Label'] = 0
                    else:
                        df_data['Label'] = 1

                    if df_log is None:
                        df_log = df_data[['Content','Label']]
                    else:
                        df_log = pd.concat([df_log,df_data[['Content','Label']]], ignore_index=True)

    log_rex = benchmark_settings['Hadoop']['regex']
    df_log['Content'] = df_log['Content'].apply(lambda x : add_var_token(log_rex,x))

    df_normal = df_log[df_log['Label']==0]
    df_abnormal = df_log[df_log['Label']==1]

    # print('Normal dataset: ',df_normal.shape)
    # print('Abnormal dataset',df_abnormal.shape)

    df_train = pd.concat([df_normal.iloc[:10000],df_abnormal.iloc[:10000]], ignore_index=True)
    df_train = df_train.sample(frac=1.0, random_state=42)


    df_test = pd.concat([df_normal.iloc[10000:13000],df_abnormal.iloc[10000:13000]],ignore_index=True)

    print('Train dataset: ',df_train.shape)
    print('Test dataset: ',df_test.shape)

    return df_train,df_test

def load_bgl_log(log_file,benchmark_settings):
    """
        读取原始的BGL日志文件，按NuLog论文中的方式划分为训练集和测试集，用于异常检测
    """
    log_format = benchmark_settings['BGL']['log_format']

    headers, regex = generate_logformat_regex(log_format)

    df_data = log_to_dataframe(os.path.join(log_file), regex, headers, log_format)

    df_log = df_data[['Content','Label']]

    log_rex = benchmark_settings['BGL']['regex']
    df_log['Content'] = df_log['Content'].apply(lambda x : add_var_token(log_rex,x))
    df_log['Label'] = df_log['Label'].apply(lambda x: 0 if x=='-' else 1)

    evaluate_len = int(len(df_log))
    
    df_train = df_log.iloc[:int(evaluate_len*0.8)]
    df_test = df_log.iloc[int(evaluate_len*0.8):evaluate_len]

    
    df_train = df_train.sample(frac=1.0, random_state=42)

    print('Normal dataset: ',df_train[df_train['Label']==0].shape)
    print('Abnormal dataset: ',df_train[df_train['Label']==1].shape)

    print('Train dataset: ',df_train.shape)
    print('Test dataset: ',df_test.shape)

    return df_train,df_test

def create_thunderbird_10m(log_file):

    linecount = 0
    max_len = 10000000

    with open(log_file, 'r', errors='ignore') as fin:
        with open('Thunderbird_10M.log','w') as n:
            for line in fin.readlines():
                n.writelines(line)
                linecount += 1
                if linecount>=max_len:
                    print("The number of lines of logs exceeds ",max_len)
                    break
    
    return

def load_thunderbird_log(log_file,benchmark_settings):
    """
        读取原始的Thunderbird日志文件，取部分前面部分划分为训练集和测试集，用于异常检测
    """
    log_format = benchmark_settings['Thunderbird']['log_format']

    headers, regex = generate_logformat_regex(log_format)

    df_data = log_to_dataframe(os.path.join(log_file), regex, headers, log_format)

    df_log = df_data[['Content','Label']]

    log_rex = benchmark_settings['Thunderbird']['regex']
    df_log['Content'] = df_log['Content'].apply(lambda x : add_var_token(log_rex,x))
    df_log['Label'] = df_log['Label'].apply(lambda x: 0 if x=='-' else 1)

    # print(df_log)
    # print(df_log.shape)

    print('Normal dataset: ',df_log[df_log['Label']==0].shape)
    print('Abnormal dataset: ',df_log[df_log['Label']==1].shape)

    evaluate_len = int(len(df_log))

    df_train = df_log.iloc[:int(evaluate_len*0.8)]
    df_test = df_log.iloc[int(evaluate_len*0.8):evaluate_len]

    df_train = df_train.sample(frac=1.0, random_state=42)

    print('Normal dataset: ',df_train[df_train['Label']==0].shape)
    print('Abnormal dataset: ',df_train[df_train['Label']==1].shape)

    print('Train dataset: ',df_train.shape)
    print('Test dataset: ',df_test.shape)

    return df_train,df_test

class Log_Dataset(Dataset):
    def __init__(self,df_train) -> None:

        # df_noraml = df_noraml.sample(frac=1.0, random_state=42)
        # df_abnormal = df_abnormal.sample(frac=1.0, random_state=42)

        # df_log = pd.concat([df_normal.iloc[:10000],df_abnormal.iloc[:10000]], ignore_index=True)
        # df_log = df_log.sample(frac=1.0, random_state=42)

        self.data = df_train

    def __getitem__(self, index):
        return self.data['Content'].iloc[index],self.data['Label'].iloc[index]

    def __len__(self):
        return len(self.data)

def load_industry_log(file_path):
    """加载日志数据
    return:
        df:完整的日志数据集
        corpus:带标签的日志数据
    """

    df_log = pd.read_csv(file_path)
    df_labeled = df_log[df_log['label_id']!=-1]
    # print(df_labeled)
    # print(len(df_labeled))
    label_count = df_labeled['label_id'].value_counts()
    # print('label_count:',label_count)
    # print(label_count)
    print('label type amount:',len(label_count))

    corpus = []
    for idx, line in df_labeled.iterrows():
        # log_temp = line['Level']+' '+line['Component'] +': '+line['Content']
        # log_temp = line['sample_raw']
        log_temp = line['Content']

        try:
            log_temp.lower()
        except:
            print('label_id:',line['label_id'])
            print('index:',idx)
            print('log:',log_temp)
            print("This preprocessed log is Null!")
        # corpus.append(log_temp.lower())
        corpus.append(log_temp)

    # for idx, line in df_log[df_log['label_id']==1661968714].iterrows():
    #     print(line["sample"])

    return df_log, corpus