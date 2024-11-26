import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.1, use_metrics=True, use_log=True):
        self.use_metrics = use_metrics
        self.use_log = use_log
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        
        self.linear_plan_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_plan_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        
        self.linear_log_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_log_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        
        self.linear_metrics_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_metrics_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout_sql = nn.Dropout(dropout)
        self.dropout_plan = nn.Dropout(dropout)
        self.dropout_log = nn.Dropout(dropout)
        self.dropout_metrics = nn.Dropout(dropout)

        model_num = 4
        if not self.use_metrics: model_num -= 1
        if not self.use_log: model_num -= 1
        self.final_linear = nn.Linear(model_dim * model_num, model_dim)

        self.edge_project = nn.Sequential(nn.Linear(model_dim, model_dim),
                                          SSP(),
                                          nn.Linear(model_dim, model_dim // 2))
        self.edge_update = nn.Sequential(nn.Linear(model_dim * 2, model_dim),
                                         SSP(),
                                         nn.Linear(model_dim, model_dim))

    def forward(self, sql, plan, log, metrics, sql_mask, plan_mask, mask=None, additional_mask=None, layer_cache=None, type=None, edge_feature=None, pair_indices=None):

        query = sql
        sql_key = sql
        sql_value = sql

        plan_key = plan
        plan_value = plan

        batch_size = query.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        sql_key_projected = self.linear_keys(sql_key)
        sql_value_projected = self.linear_values(sql_value)
        query_projected = self.linear_query(query)
        sql_key_shaped = shape(sql_key_projected)
        sql_value_shaped = shape(sql_value_projected)

        plan_key_projected = self.linear_plan_keys(plan_key)
        plan_value_projected = self.linear_plan_values(plan_value)
        plan_key_shaped = shape(plan_key_projected)
        plan_value_shaped = shape(plan_value_projected)

        query_shaped = shape(query_projected)
        query_len = query_shaped.size(2)
        sql_key_len = sql_key_shaped.size(2)
        plan_key_len = plan_key_shaped.size(2)

        # sql encoder
        sql_query_shaped = query_shaped / math.sqrt(dim_per_head)
        scores = torch.matmul(sql_query_shaped, sql_key_shaped.transpose(2, 3))
        top_score = scores.view(batch_size, scores.shape[1],
                                query_len, sql_key_len)[:, 0, :, :].contiguous()
        attn = self.softmax(scores)
        drop_attn = self.dropout_sql(attn)
        context = torch.matmul(drop_attn, sql_value_shaped)
        sql_context = unshape(context)

        # plan encoder
        sql_query_shaped = query_shaped / math.sqrt(dim_per_head)
        scores = torch.matmul(sql_query_shaped, plan_key_shaped.transpose(2, 3))
        attn = self.softmax(scores)
        drop_attn = self.dropout_plan(attn)
        context = torch.matmul(drop_attn, plan_value_shaped)
        plan_context = unshape(context)

        # metrics encoder
        if self.use_metrics:
            metrics = metrics.unsqueeze(1)
            metrics_key = metrics
            metrics_value = metrics

            metrics_key_projected = self.linear_metrics_keys(metrics_key)
            metrics_value_projected = self.linear_metrics_values(metrics_value)
            metrics_key_shaped = shape(metrics_key_projected)
            metrics_value_shaped = shape(metrics_value_projected)

            metrics_key_len = metrics_key_shaped.size(2)


            sql_query_shaped = query_shaped / math.sqrt(dim_per_head)
            scores = torch.matmul(sql_query_shaped, metrics_key_shaped.transpose(2, 3))
            attn = torch.sigmoid(scores)
            drop_attn = self.dropout_metrics(attn)
            context = torch.matmul(drop_attn, metrics_value_shaped)
            metrics_context = unshape(context)


        if self.use_log:
            log = log.unsqueeze(1)
            log_key = log
            log_value = log

            log_key_projected = self.linear_log_keys(log_key)
            log_value_projected = self.linear_log_values(log_value)
            log_key_shaped = shape(log_key_projected)
            log_value_shaped = shape(log_value_projected)

            sql_query_shaped = query_shaped / math.sqrt(dim_per_head)
            scores = torch.matmul(sql_query_shaped, log_key_shaped.transpose(2, 3))
            attn = torch.sigmoid(scores) 
            drop_attn = self.dropout_log(attn)
            context = torch.matmul(drop_attn, log_value_shaped)
            log_context = unshape(context)

        context = torch.cat([sql_context, plan_context], dim=-1)

        if self.use_metrics:
            context = torch.cat([context, metrics_context], dim=-1)
        if self.use_log:
            context = torch.cat([context, log_context], dim=-1)

        output = self.final_linear(context)

        return output, top_score


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


