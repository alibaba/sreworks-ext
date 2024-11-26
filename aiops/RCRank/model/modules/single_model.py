import torch
from torch import nn
from model.modules.QueryFormer.QueryFormer import QueryFormer
from model.modules.LogModel.log_model import LogModel

# ----------------------------------------- 单模态 -----------------------------------

class SQLOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.activation = nn.ReLU()        

        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        
        sql_emb = sql_emb.last_hidden_state

        sql_emb = sql_emb[:, 0, :]
        sql_emb = self.sql_last_emb(sql_emb)
        sql_emb = self.activation(sql_emb)
        
        pred_label = self.pred_label_cross(sql_emb)
        pred_opt = self.pred_opt_cross(sql_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    

class PlanOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)

        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        self.activation = nn.ReLU()        
        
        self.init_params()
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        plan_emb = self.plan_model(plan)

        plan_emb = plan_emb[:, 0, :]
        plan_emb = self.activation(plan_emb)

        pred_label = self.pred_label_cross(plan_emb)
        pred_opt = self.pred_opt_cross(plan_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt


class LogOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        self.activation = nn.ReLU()        
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        log_emb = self.log_model(log)
        log_emb = self.activation(log_emb)
        
        pred_label = self.pred_label_cross(log_emb)
        pred_opt = self.pred_opt_cross(log_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt


class TimeOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.time_model = time_model
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        self.time_tran_emb = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.ReLU()        
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb)
        
        time_emb = torch.flatten(time_emb, start_dim=1)
        time_emb = self.time_tran_emb(time_emb)
        time_emb = self.activation(time_emb)
        
        pred_label = self.pred_label_cross(time_emb)
        pred_opt = self.pred_opt_cross(time_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
# ----------------------------------------- 单模态 -----------------------------------

# Concat
class ConcatOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim, emb_dim)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        

        self.cross_mean = cross_mean
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        sql_emb = sql_emb.last_hidden_state
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb)
        
        sql_emb = sql_emb[:, 0, :]
        sql_emb = self.sql_last_emb(sql_emb)

        plan_emb = plan_emb[:, 0, :]
        time_emb = torch.flatten(time_emb, start_dim=1)
        time_emb = self.time_tran_emb(time_emb)

        emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        pred_label = self.pred_label_concat(emb)
        pred_opt = self.pred_opt_concat(emb)
        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt

