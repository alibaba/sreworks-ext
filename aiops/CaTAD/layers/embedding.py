import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, unit='min'):
        super(TemporalEmbedding, self).__init__()

        second_size = 60
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
            
        self.hour_embed = FixedEmbedding(hour_size, d_model)
        self.weekday_embed = FixedEmbedding(weekday_size, d_model)
        self.day_embed = FixedEmbedding(day_size, d_model)
        self.month_embed = FixedEmbedding(month_size, d_model)
        self.minute_embed = FixedEmbedding(minute_size, d_model)
        self.unit = unit
        if unit == 's':
            self.second_embed = FixedEmbedding(second_size, d_model)

    def forward(self, x):
        x = x.long()
        month_x = self.month_embed(x[:, :, 0])
        day_x = self.day_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        hour_x = self.hour_embed(x[:, :, 3])
        minute_x = self.minute_embed(x[:, :, 4])
        if self.unit == 's':
            second_x = self.second_embed(x[:, :, 5])
        else:
            second_x = 0
        return hour_x + weekday_x + day_x + month_x + minute_x + second_x
        
        
        
class Embedding(nn.Module):
    def __init__(self, c_in, d_model, freq='h', dropout=0.1):
        super(Embedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, unit=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # if x_mark is None:
        #     x = self.value_embedding(x) + self.position_embedding(x)
        # else:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)