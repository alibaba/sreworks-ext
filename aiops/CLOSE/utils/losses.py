import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from typing import Iterable, Dict
import torch

def calculate_center(model,all_event_log):

    event_center = {}

    for event in all_event_log:
        corpus = all_event_log[event]

        total_embeddings = 0

        with torch.no_grad():
            embeddings = model.encode(corpus,convert_to_numpy=False, convert_to_tensor=True,normalize_embeddings=True).clone()
            total_embeddings += torch.sum(embeddings, dim=0)

        center = total_embeddings/len(corpus)
        event_center[event] = center

    return event_center

class MNR_Hyper_Loss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20, similarity_fct = util.cos_sim, hyper_ratio = 0.01, log_to_event={}, event_center={}):
        super(MNR_Hyper_Loss, self).__init__()
        self.model = model
        self.scale = scale 
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.hyper_ratio = hyper_ratio
        self.log_to_event = log_to_event
        self.event_center = event_center


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # print(sentence_features)
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # print("reps:",len(reps))
        embeddings_a = reps[0] #(b,768)
        # print("embeddings_a:",embeddings_a.size())
        embeddings_b = torch.cat(reps[1:]) # (b,768)
        # print("embeddings_b:",embeddings_b.size())

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale #(b,b)
        # print("scores:",scores.size())
        MNR_labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device) 
        
        MNR_loss = self.cross_entropy_loss(scores, MNR_labels)

        if self.hyper_ratio==0:
            return MNR_loss
        
        center_embeddings = []

        for b in range(len(sentence_features[0]['input_ids'])):
            log_token = sentence_features[0]['input_ids'][b]
            token_mask = sentence_features[0]['attention_mask'][b]
            log_token = log_token.cpu().numpy()
            token_mask = token_mask.cpu().numpy()
            log_token = log_token[token_mask!=0]
            event_id = self.log_to_event[tuple(log_token.tolist())]
            center = self.event_center[event_id].unsqueeze(dim=0)
            center_embeddings.append(center)

        center_embeddings = torch.cat(center_embeddings,dim=0)
        
        hyper_similarity = torch.cosine_similarity(embeddings_a,center_embeddings)
        hyper_labels = torch.ones_like(hyper_similarity,device=scores.device)
        
        # hyper_loss = self.mse_loss(embeddings_a,center_embeddings)
        hyper_loss = self.mse_loss(hyper_similarity,hyper_labels)
        
        loss = MNR_loss + self.hyper_ratio*hyper_loss
        
        return loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}