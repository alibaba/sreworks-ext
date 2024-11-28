import torch.nn as nn
import torch

from pretrain.alignment_new.pretrain import Alignment
from model.modules.LogModel.log_model import LogModel


class Predict(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, ff_dim, dropout=0.1):
        super(Predict, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.input_fc = nn.Linear(input_dim, model_dim)  

    def forward(self, x):
        x = self.input_fc(x)  
        x = self.transformer_encoder(x)
        return x

    
class GateComDiffPretrainModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()
        
        self.time_model = time_model
        self.log_model = LogModel(1, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.plan_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)

        self.common_cross_model = cross_model
        self.rootcause_cross_model = rootcause_cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for _ in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)

        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
        
        self.init_params()
        self.device = device
        self.alignmentModel = Alignment(device=device)
        self.alignmentModel.load_state_dict(torch.load('./pretrain/alignment_new/model30.pth')) # now use
        
        self.sql_model = sql_model
        self.log_model = self.alignmentModel.log_model
        self.plan_model = self.alignmentModel.plan_model
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
            
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
                
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb)
        time_emb = time_emb.squeeze()

        sql_emb = self.sql_last_emb(sql_emb)
        
        common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        
        for i in range(5):
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            time_emb_tmp = self.gate_metrics[i](time_emb)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb
            
            emb = self.rootcause_cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
        
            emb = emb.mean(dim=1)
            emb = common_emb + emb
            
            pred_label = self.pred_label_cross_list[i](emb)
            pred_opt = self.pred_opt_cross_list[i](emb)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
        return pred_label_output, pred_opt_output
    
    
