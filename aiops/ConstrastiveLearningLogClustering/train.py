import torch
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.losses import *
from utils.evaluation import *

seed = 22
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed) 
random.seed(seed) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model_name = 'multi-qa-MiniLM-L6-cos-v1'
batch_size = 20

evaluate_score = []

for test_name in benchmark_settings:
    print("Test dataset: ", test_name)
    test_log_type = test_name

    train_len = 20000
    
    train_examples = generate_samples(train_len,test_log_type,batch_size)

    print("Train sentence pairs: ",len(train_examples))

    model = SentenceTransformer(model_name)

    special_tokens = ['[var]']
    word_embedding_model = model._first_module()
    word_embedding_model.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    ft_train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=batch_size)
    all_event_log, log_to_event = load_event_log(test_log_type=test_log_type,benchmark_settings=benchmark_settings,model=model)
    event_center = calculate_center(model,all_event_log)
    ft_train_loss = MNR_Hyper_Loss(model,log_to_event=log_to_event,event_center=event_center,hyper_ratio=0)
    model.fit(train_objectives=[(ft_train_dataloader, ft_train_loss)], epochs=2, warmup_steps=100, scheduler='constantlr', optimizer_params={'lr': 3e-5})

    for _ in range(5):
        train_examples = generate_samples(train_len,test_log_type)
        
        ft_train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=batch_size)
        all_event_log, log_to_event = load_event_log(test_log_type=test_log_type,benchmark_settings=benchmark_settings,model=model)
        event_center = calculate_center(model,all_event_log)
        ft_train_loss = MNR_Hyper_Loss(model,log_to_event=log_to_event,event_center=event_center,hyper_ratio=0.2)
        model.fit(train_objectives=[(ft_train_dataloader, ft_train_loss)], epochs=1, warmup_steps=0, scheduler='constantlr', optimizer_params={'lr': 1e-5})

    print("Model sentence pairs trainning done.")
    
    model.to('cpu')

    df_log, test_corpus = load_test_log(test_log_type,benchmark_settings)
    corpus_embeddings = generate_embeddings(model,test_corpus)
    
    distance_threshold = benchmark_settings[test_log_type]['distance_threshold']
    
    clustered_sentences, cluster_assignment = embeddings_clustering(test_corpus, corpus_embeddings, distance_threshold)
    score, event_amount, cluster_amount = clustering_evaluate(test_log_type, cluster_assignment, clustered_sentences)
    
    score['dataset'] = test_log_type
    score['event amount'] = event_amount
    score['cluster amount'] = cluster_amount

    evaluate_score.append(score)

df_socre = pd.DataFrame(evaluate_score)


