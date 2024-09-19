import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SpatialAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_features)
        self.key = nn.Linear(in_features, hidden_features)
        self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (n_sample, n_node, t, d)
        x = x.permute(0,2,1,3)
        q = self.query(x) # Query: (n_sample, n_node, t, hidden_features)
        k = self.key(x).transpose(-2, -1) # Key: (n_sample, n_node, hidden_features, t)
        v = self.value(x) # Value: (n_sample, n_node, t, d)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k) # (n_sample, n_node, t, t)
        attn_scores = self.softmax(attn_scores) # Normalize scores
        
        # Apply attention scores to values
        attn_output = torch.matmul(attn_scores, v) # (n_sample, n_node, t, d)
        return attn_output.permute(0,2,1,3)

class TemporalAttention(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_features)
        self.key = nn.Linear(in_features, hidden_features)
        self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (n_sample, n_node, t, d)
        q = self.query(x) # Query: (n_sample, n_node, t, hidden_features)
        k = self.key(x).transpose(-2, -1) # Key: (n_sample, n_node, hidden_features, t)
        v = self.value(x) # Value: (n_sample, n_node, t, d)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k) # (n_sample, n_node, t, t)
        attn_scores = self.softmax(attn_scores) # Normalize scores
        
        # Apply attention scores to values
        attn_output = torch.matmul(attn_scores, v) # (n_sample, n_node, t, d)
        return attn_output


class PeriodFusionAttn(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(PeriodFusionAttn, self).__init__()
        self.key = nn.Linear(in_features, hidden_features)
        self.query = nn.Linear(in_features, hidden_features)
        self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, t, h_d, p = x.size()  # x [batch_size, seq_len, hid_dim, n_period]
        x_flat = x.permute(0, 3, 1, 2).contiguous().view(b, p, -1)  
        
        k = self.key(x_flat).transpose(-2, -1)
        q = self.query(x_flat)
        v = self.value(x_flat)

        attn_scores = torch.matmul(q, k) 
        attn_scores = self.softmax(attn_scores)
        
        fused = torch.matmul(attn_scores, v)
        fused = fused.view(b, p, t, h_d).permute(0, 2, 3, 1)  
        fused = torch.mean(fused, dim=-1, keepdim=False)
        return fused

class PeriodFusionAttn_case(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(PeriodFusionAttn_case, self).__init__()
        self.key = nn.Linear(in_features, hidden_features)
        self.query = nn.Linear(in_features, hidden_features)
        self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, dataset, epoch, name):
        b, t, h_d, p = x.size()  # x [batch_size, seq_len, hid_dim, n_period]
        x_flat = x.permute(0, 3, 1, 2).contiguous().view(b, p, -1)  
        
        k = self.key(x_flat).transpose(-2, -1)
        q = self.query(x_flat)
        v = self.value(x_flat)

        attn_scores = torch.matmul(q, k) 
        attn_scores = self.softmax(attn_scores)
        
        fused = torch.matmul(attn_scores, v)
        fused = fused.view(b, p, t, h_d).permute(0, 2, 3, 1)  
        fused = torch.mean(fused, dim=-1, keepdim=False)
        return fused