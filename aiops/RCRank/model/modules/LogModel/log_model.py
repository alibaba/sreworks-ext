import torch
from torch import nn
from torch.nn import functional as F


class LogModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        
        self.ll_1 = nn.Linear(input_dim, hidden_dim)
        self.ll_2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, output_dim)

        self.init_params()
            
    def forward(self, input_ids):
        output = self.ll_1(input_ids)
        output = F.relu(output)
        output = self.ll_2(output)
        output = F.relu(output)
        output = self.cls(output)

        return output
    
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)