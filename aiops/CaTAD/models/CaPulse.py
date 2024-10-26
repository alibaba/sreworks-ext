import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.attn import SpatialAttention, TemporalAttention, PeriodFusionAttn
from layers.NF import PeriodNF
from layers.embedding import Embedding
from layers.conv import Inception_Block
import torch
from utils.helper import FFT_for_Period, causal_interve

class MultiPeriodEncoder(nn.Module):
    def __init__(self, seq_len, top_k, d_model, d_ff, num_kernels, num_causes):
        super(MultiPeriodEncoder, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(d_ff, d_model, num_kernels=num_kernels)
        )
        self.linear_proj = nn.Linear(seq_len, num_causes)
        self.num_causes = num_causes
    def forward(self, input):
        '''
        refer to Time-Series-Library https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
        '''
        b, t, n = input.size()
        period_list, weight = FFT_for_Period(input, self.k) # find the local period by fft

        feat_pyramid = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len  % period != 0:
                length = ( (self.seq_len  // period) + 1) * period
                padding = torch.zeros([input.shape[0], (length - self.seq_len ), input.shape[2]]).to(input.device)
                out = torch.cat([input, padding], dim=1)
            else:
                length = self.seq_len 
                out = input
            # reshape
            out = out.reshape(b, length // period, period, n).permute(0, 3, 1, 2).contiguous() # out [512, 119, 8] --> [512, 1, 60, 8]
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(b, -1, n)
            out = out[:, :self.seq_len , :] #(b, t, d)
            out = self.linear_proj(out.permute(0,2,1)).permute(0,2,1) #(b, n_causes, dim_cause)
            feat_pyramid.append(out)
        feat_pyramid = torch.stack(feat_pyramid, dim=-1)

        weight = F.softmax(weight, dim=1)
        weight = weight.unsqueeze(1).unsqueeze(1).repeat(1, self.num_causes, n, 1)
        return feat_pyramid, weight

class MultiPeriodFusion(nn.Module):
    def __init__(self, hidden_size,n_causes):
        super(MultiPeriodFusion, self).__init__()
        self.adaptive_fusion = PeriodFusionAttn(hidden_size*n_causes, hidden_size)

    def forward(self, feat_pyramid, weight):
        feat1 = torch.sum(feat_pyramid * weight, -1)
        feat2 = self.adaptive_fusion(feat_pyramid)
        output = 0.5*feat1 + 0.5*feat2
        return output

class CaPulse(nn.Module):

    def __init__ (self, n_blocks, input_size, hidden_size, n_layers ,dropout, batch_norm, n_node, interve_level, period, unit, use_multidim, n_causes):
        super(CaPulse, self).__init__()
        self.top_k = 3
        self.d_ff = 16
        self.num_kernels = 2
        self.n_node=n_node
        self.n_blocks=n_blocks
        self.n_layers=n_layers
        self.seq_len = 60
        self.interve_level = interve_level
        self.n_causes = n_causes

        self.embedding = Embedding(c_in=self.n_node, d_model=hidden_size, freq=unit, dropout=dropout)
        self.feat_ext = MultiPeriodEncoder(seq_len=self.seq_len, top_k = self.top_k, d_model=hidden_size, d_ff=self.d_ff, num_kernels=self.num_kernels, num_causes=n_causes)
        self.feat_fusion = MultiPeriodFusion(hidden_size,n_causes)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, self.n_node*hidden_size, bias=True)
        self.ST_ext = nn.ModuleList([
                                    nn.Sequential(
                                        SpatialAttention(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        TemporalAttention(hidden_size, hidden_size),
                                        nn.ReLU()
                                    )
                                    for _ in range(self.n_layers)])
        self.use_multidim = use_multidim
        if use_multidim:
            self.proj4nf = nn.Linear(n_node*n_causes, self.seq_len, bias=True)
            self.period_nf = PeriodNF(n_blocks, n_node, hidden_size, n_layers, period = period, cond_label_size=hidden_size, batch_norm=batch_norm)
        else:
            self.proj4nf = nn.Linear(n_node*n_causes, self.seq_len*n_node, bias=True)
            self.period_nf = PeriodNF(n_blocks, 1, hidden_size, n_layers, period = period, cond_label_size=hidden_size, batch_norm=batch_norm)

    def forward(self, x, x_time):
        neg_log_prob, cause, x_enc, x_aug = self.test(x, x_time)
        return neg_log_prob.mean(), cause, x_enc, x_aug

    def test(self, x, x_time):
        # x: (n_sample, n_node, t, d)
        n_sample, n_node, t, d = x.shape

        x_enc = x.permute(0,2,1,3).squeeze() # x: (n_sample, t, n)
        x_aug = causal_interve(x_enc, self.interve_level) # causal intervention

        ##### local multi-periods feature capture ######
        # embedding
        enc_emb = self.embedding(x_enc, x_time)  # (n_sample, t, d_model)
        aug_emb = self.embedding(x_aug, x_time)

        # feature capture
        enc_pyramid, enc_weight = self.feat_ext(enc_emb)
        aug_pyramid, aug_weight = self.feat_ext(aug_emb)

        enc_out = self.layer_norm(self.feat_fusion(enc_pyramid, enc_weight)) # (n_sample, n_causes, dim_cause)
        aug_out = self.layer_norm(self.feat_fusion(aug_pyramid, aug_weight))
        enc_out_proj = self.proj(enc_out + aug_out) # (n_sample, t, d_model) --> (n_sample, t, n_node * h_d)

        h = enc_out_proj.reshape(n_sample, self.n_causes, n_node, -1).permute(0,2,1,3) # (b, n_node, n_causes, h_d) 
        for layer in self.ST_ext:
            h = layer(h) # (b, n_node, n_causes, h_d) 


        if self.use_multidim:
            x_nf = x.squeeze().permute(0,2,1).reshape(-1, n_node)
            h_nf = self.proj4nf(h.permute(0,3,1,2).reshape(n_sample,-1,n_node*self.n_causes)).squeeze().permute(0,2,1).reshape(-1,h.shape[3])
        else:
            x_nf = x.reshape((-1,d))
            h_nf = self.proj4nf(h.permute(0,3,1,2).reshape(n_sample,-1,n_node*self.n_causes)).squeeze().permute(0,2,1).reshape(-1,h.shape[3])

        log_prob = self.period_nf.log_prob(x_nf, h_nf).reshape([n_sample,-1])
        log_prob = log_prob.mean(dim=1)

        return -log_prob, torch.mean(h, 1).reshape(n_sample, self.n_causes, -1), enc_out, aug_out

