from backbone.chat import Qwen, OpenaiChat
from functools import partial
import copy
import json 
import os
import pickle
import networkx as nx
MODELS = {
    "qwen-max": Qwen,
    "qwen-turbo": Qwen,
    "qwen-7B":OpenaiChat,
    "qwen-14B":OpenaiChat,
    "llama3": OpenaiChat,
    "mistral-7B": OpenaiChat,
    "qwen":Qwen
}

root_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load_model_by_config_name(model='qwen'):
    with open(os.path.join(root_path, f"common/configs/backbone.json"),'r') as f:
        model_config = json.load(f)['qwen']
    model_config['llm_name'] = model
    return MODELS[model](**model_config)


def save_data(df, path, file_name):
    with open(f"{path}/{file_name}.pkl", "wb") as f:
            pickle.dump(df, f)

def load_data( path, file_name):
    with open(f"{path}/{file_name}.pkl", "rb") as f:
            data = pickle.load(f)
    return data

def load_system_info(path='data/mc/description.json'):
    nodes = []
    edges = []
    with open(path, 'r') as f:
        data = json.load(f)['MC']
        for node in data['service description'].keys():
            nodes.append((node, {'description':data['service description'][node]}))
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(data['interactions'])
    G.graph['system introduction'] = data['system description']
    return G

    