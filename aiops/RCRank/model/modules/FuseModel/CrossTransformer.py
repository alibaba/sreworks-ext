import torch
from torch import nn

import torch
import torch.nn as nn

from .Attention import PositionwiseFeedForward, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, plan, log, metrics, src_mask, plan_mask):
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, plan, log, metrics, src_mask, plan_mask)

        out = self.dropout(context) + inputs
        return self.feed_forward(out), attn

class CrossTransformer(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, attn_modules):
        super(CrossTransformer, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)


    def forward(self, sql, plan, log, metrics, sql_mask, plan_mask):

        out = sql

        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, plan, log, metrics, sql_mask, plan_mask)

        out = self.layer_norm(out)
        return out

