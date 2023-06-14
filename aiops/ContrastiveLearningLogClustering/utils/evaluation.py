from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_embeddings(model,corpus):

    #Mean Pooling - Take average of all tokens
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    #Encode text
    # def encode(texts):
    #     # Tokenize sentences
    #     encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    #     # encoded_input.cuda()

    #     # Compute token embeddings
    #     with torch.no_grad():
    #         model_output = model(**encoded_input, return_dict=True)

    #     # Perform pooling
    #     embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    #     # Normalize embeddings
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
        
    #     return embeddings

    # embedder = SentenceTransformer(model_name)
    # print(embedder)

    lower_corpus = [s.lower() for s in corpus]
    # 生成日志embedding
    start_time = int(time.time())
    # corpus_embeddings = embedder.encode(lower_corpus)
    corpus_embeddings = model.encode(lower_corpus,normalize_embeddings=True)
    # Normalize the embeddings to unit length
    # corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    return corpus_embeddings

def embeddings_clustering(corpus, corpus_embeddings, distance_threshold=0.1):

    def compute_dot_similarity(a):
        score = a.dot(a.transpose(1,0)) #(b,n)*(n*b)
        return score #(b*b)

    clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='single', distance_threshold=distance_threshold)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])


    return clustered_sentences, cluster_assignment

def clustering_evaluate(log_type, cluster_assignment, clustered_sentences):
    
    label_true = []
    
    if log_type == "Flink" or log_type == 'ODPS':
        df_log = pd.read_csv(log_type.lower()+'_test.csv')
        # df_groundtruth = df_log_structured['EventId']
        df_log = df_log[df_log['label_id']!=-1]
        label_count = df_log['label_id'].value_counts()
        event_amount = len(label_count)
        cluster_amount = len(clustered_sentences)
        print('event amount: ',event_amount)
        print('cluster amount: ',cluster_amount)

        for idx, line in df_log.iterrows():
            label = line['label_id']
            label_true.append(label)
    else:
        df_log_structured = pd.read_csv("./logs/"+log_type+"/"+log_type+"_2k.log_structured.csv")
        # df_groundtruth = df_log_structured['EventId']
        label_count = df_log_structured['EventId'].value_counts()
        event_amount = len(label_count)
        cluster_amount = len(clustered_sentences)
        print('event amount: ',event_amount)
        print('cluster amount: ',cluster_amount)

        for idx, line in df_log_structured.iterrows():
            label = line['EventId']
            label_true.append(int(label[1:])-1)

    rand_index = metrics.rand_score(label_true, cluster_assignment) 
    homogeneity = metrics.homogeneity_score(label_true, cluster_assignment)#
    completeness = metrics.completeness_score(label_true, cluster_assignment)
    v_measure = metrics.v_measure_score(label_true,cluster_assignment, beta=1) #v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    adj_rand_index = metrics.adjusted_rand_score(label_true, cluster_assignment)
    normalized_mi = metrics.normalized_mutual_info_score(label_true, cluster_assignment)
    
    # print("rand_index: ",rand_index)
    # print('homogeneity score: ',homogeneity) 
    # print('completeness score: ',completeness) 
    # print('v measure score: ',v_measure)
    print('ARI',adj_rand_index)
    print('NMI',normalized_mi)

    # df_log_structured.iterrows
    label_groundtrue = label_true
    # for idx, line in df_log_structured.iterrows():
    #     label_groundtrue.append(int(line['EventId'][1:])-1)

    series_parsedlog = pd.Series(cluster_assignment)
    series_groundtruth = pd.Series(label_groundtrue)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    # series_groundtruth_valuecounts = series_groundtruth.value_counts()

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False

    parsing_accuracy = float(accurate_events) / series_groundtruth.size
    
    print("parsing accuarcy: ",parsing_accuracy)

    # F1_measure = metrics.f1_score(label_true,label_pre,average='micro')
    # print("F1 score: ",F1_measure)

    score = {}
    score['rand index'] = rand_index
    score['parsing accuracy'] = parsing_accuracy
    score['homogeneity'] = homogeneity
    score['completeness'] = completeness
    score['v measure'] = v_measure
    score['ARI'] = adj_rand_index
    score['NMI'] = normalized_mi

    return score, event_amount, cluster_amount

def clustering_evaluate_industry(file_name, cluster_assignment, clustered_sentences):

    df_log = pd.read_csv(file_name)
    # df_groundtruth = df_log_structured['EventId']
    df_log = df_log[df_log['label_id']!=-1]
    label_count = df_log['label_id'].value_counts()
    event_amount = len(label_count)
    cluster_amount = len(clustered_sentences)
    print('event amount: ',event_amount)
    print('cluster amount: ',cluster_amount)
    
    label_true = []

    for idx, line in df_log.iterrows():
        label = line['label_id']
        label_true.append(label)

    rand_index = metrics.rand_score(label_true, cluster_assignment) 
    homogeneity = metrics.homogeneity_score(label_true, cluster_assignment)
    completeness = metrics.completeness_score(label_true, cluster_assignment)
    v_measure = metrics.v_measure_score(label_true,cluster_assignment, beta=1) #v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    adj_rand_index = metrics.adjusted_rand_score(label_true, cluster_assignment)
    normalized_mi = metrics.normalized_mutual_info_score(label_true, cluster_assignment)
    
    # print("rand_index: ",rand_index)
    # print('homogeneity score: ',homogeneity) 
    # print('completeness score: ',completeness) 
    # print('v measure score: ',v_measure)
    print('ARI',adj_rand_index)
    print('NMI',normalized_mi)

    label_groundtrue = label_true

    series_parsedlog = pd.Series(cluster_assignment)
    series_groundtruth = pd.Series(label_groundtrue)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    # series_groundtruth_valuecounts = series_groundtruth.value_counts()

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False

    parsing_accuracy = float(accurate_events) / series_groundtruth.size
    
    print("parsing accuarcy: ",parsing_accuracy)

    # F1_measure = metrics.f1_score(label_true,label_pre,average='micro')
    # print("F1 score: ",F1_measure)

    score = {}
    score['rand index'] = rand_index
    score['parsing accuracy'] = parsing_accuracy
    score['homogeneity'] = homogeneity
    score['completeness'] = completeness
    score['v measure'] = v_measure
    score['ARI'] = adj_rand_index
    score['NMI'] = normalized_mi

    return score, event_amount, cluster_amount
