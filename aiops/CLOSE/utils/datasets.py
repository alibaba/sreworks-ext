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
from sentence_transformers import InputExample

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

    df_log = None

    for log_type in benchmark_settings:
        if log_type != test_log_type:
            log_format = benchmark_settings[log_type]['log_format']

            headers, regex = generate_logformat_regex(log_format)
            df_data = log_to_dataframe(os.path.join("./logs/"+log_type+"/", log_type+"_2k.log"), regex, headers, log_format)
            
            log_rex = benchmark_settings[log_type]['regex']
            df_data['Content'] = df_data['Content'].apply(lambda x : add_var_token(log_rex,x))
            
            if df_log is None:
                df_log = df_data['Content']
            else:
                # df_log.append(df_data['Content'])
                df_log = pd.concat([df_log,df_data['Content']], ignore_index=True)

        
        # for idx, line in df_log.iterrows():
        #     # log_temp = line['Level']+' '+line['Component'] +': '+line['Content']
        #     log_temp = line['Content']
        #     # corpus.append(log_temp.lower())
        #     corpus.append(log_temp)

    return df_log

def load_test_log(log_type,benchmark_settings):

    # if log_type=="HDFS":
    #     log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    # elif log_type=="Windows":
    #     log_format = '<Date> <Time> <Level> <Component> <Content>'
    # elif log_type=="HPC":
    #     log_format = '<Logld> <Node> <Component> <State> <Time> <Flag> <Content>'

    log_format = benchmark_settings[log_type]['log_format']

    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(os.path.join("./logs/"+log_type+"/", log_type+"_2k.log"), regex, headers, log_format)
    
    log_rex = benchmark_settings[log_type]['regex']
    df_log['Content'] = df_log['Content'].apply(lambda x : (add_var_token(log_rex,x)))

    corpus = []
    for idx, line in df_log.iterrows():
        # log_temp = line['Level']+' '+line['Component'] +': '+line['Content']
        temp_log = line['Content']
        # corpus.append(log_temp.lower())

        corpus.append(temp_log)

    # log_rex = benchmark_settings[log_type]['regex']
    # corpus = [add_var_token(log_rex,s) for s in corpus]

    return df_log, corpus


def generate_positive_samples(test_log_type=None, benchmark_settings=None):

    positive_samples = {}

    for log_type in benchmark_settings:
        if log_type != test_log_type and log_type!='industrial1' and log_type!='industrial2':
            
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
            
    if test_log_type=='industrial1' or test_log_type=='industrial2':
        
        log_type = 'industrial1' if test_log_type=='industrial2' else 'industrial2'
        # print("Loading",log_type,"positive pairs...")
        
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

    all_event = OrderedDict()

    for d in positive_samples:
        # print(d)
        for e in positive_samples[d]:
            all_event[(d,e)] = len(positive_samples[d][e])
            for pairs in positive_samples[d][e]:
            # print(i)
                positive_corpus.append(pairs)
    
    return positive_corpus, all_event, positive_samples


def generate_neutral_samples(test_log_type=None, positive_corpus=[], benchmark_settings={}):

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

    return list(neutral_corpus)

def generate_negetive_samples(test_log_type=None, positive_corpus=[], neutral_corpus=[], benchmark_settings={}):

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

    return list(negetive_corpus)

def generate_contrastive_samples(positive_samples, all_event, batch_size, max_len=100000):

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

    return contrastive_corpus

def generate_contrastive_samples2(positive_samples, all_event, batch_size, max_len=100000):

    contrastive_corpus = []

    max_len = max_len
    
    event_list = list(all_event.keys())
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

    return contrastive_corpus

def industry_positive_samples(log_path,batch_size):

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

    return positive_corpus, all_event, positive_samples

def load_event_log(test_log_type=None, benchmark_settings=None, model=None):

    all_event_log = {}
    log_to_event = {}

    for log_type in benchmark_settings:
        if log_type != test_log_type and log_type!='industrial1' and log_type!='industrial2':
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

    if test_log_type=='industrial1' or test_log_type=='industrial2':
    
        log_type = 'industrial1' if test_log_type=='industrial2' else 'industrial2'
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

def load_industry_log(file_path):

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

def generate_samples(sample_len,test_log_type,batch_size):
    
    positive_corpus, all_event, positive_samples = generate_positive_samples(test_log_type=test_log_type,benchmark_settings=benchmark_settings)
    contrastive_corpus = generate_contrastive_samples(positive_samples,all_event,batch_size,max_len=sample_len)

    random_index = [i for i in range(len(contrastive_corpus)//batch_size)]

    random.shuffle(random_index)

    train_corpus = []

    for i in random_index:
        train_corpus.append(contrastive_corpus[i*batch_size:i*batch_size+batch_size])

    train_examples = []
    for batch_corpus in train_corpus:
        for pairs in batch_corpus:
            train_examples.append(InputExample(texts=list(pairs)))
            
    return train_examples

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