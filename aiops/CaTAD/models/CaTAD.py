
#%%
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.attn import SpatialAttention, TemporalAttention, PeriodFusionAttn
from layers.NF import PeriodNF
from layers.embedding import Embedding
from layers.conv import Inception_Block
import torch
from utils.helper import vector_aug, FFT_for_Period, causal_interve

class MultiPeriodEncoder(nn.Module):
    def __init__(self, seq_len, top_k, d_model, d_ff, num_kernels):
        super(MultiPeriodEncoder, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(d_ff, d_model, num_kernels=num_kernels)
        )
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
            out = out.reshape(b, length // period, period, n).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(b, -1, n)
            feat_pyramid.append(out[:, :self.seq_len , :])
        feat_pyramid = torch.stack(feat_pyramid, dim=-1)
        weight = F.softmax(weight, dim=1)
        weight = weight.unsqueeze(1).unsqueeze(1).repeat(1, t, n, 1)
        return feat_pyramid, weight

class MultiPeriodFusion(nn.Module):
    def __init__(self, hidden_size):
        super(MultiPeriodFusion, self).__init__()
        self.adaptive_fusion = PeriodFusionAttn(hidden_size*60, hidden_size)

    def forward(self, feat_pyramid, weight):
        feat1 = torch.sum(feat_pyramid * weight, -1)
        feat2 = self.adaptive_fusion(feat_pyramid)
        output = 0.5*feat1 + 0.5*feat2
        return output

class CaTAD(nn.Module):

    def __init__ (self, n_blocks, input_size, hidden_size, n_layers ,dropout, batch_norm, n_node, interve_level, period, unit, use_multidim):
        super(CaTAD, self).__init__()
        self.top_k = 3
        self.d_ff = 16
        self.num_kernels = 2
        self.n_node=n_node
        self.n_blocks=n_blocks
        self.n_layers=n_layers
        self.seq_len = 60
        self.interve_level = interve_level
        
        self.embedding = Embedding(c_in=self.n_node, d_model=hidden_size, freq=unit, dropout=dropout)
        self.feat_ext = MultiPeriodEncoder(seq_len=self.seq_len, top_k = self.top_k, d_model=hidden_size, d_ff=self.d_ff, num_kernels=self.num_kernels)
        self.feat_fusion = MultiPeriodFusion(hidden_size)
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
            self.proj4nf = nn.Linear(n_node, 1, bias=True)
            self.period_nf = PeriodNF(n_blocks, n_node, hidden_size, n_layers, period = period, cond_label_size=hidden_size, batch_norm=batch_norm)
        else:
            self.period_nf = PeriodNF(n_blocks, 1, hidden_size, n_layers, period = period, cond_label_size=hidden_size, batch_norm=batch_norm)

    def forward(self, x, x_time):
        neg_log_prob, cause, x_enc, x_aug = self.test(x, x_time)
        return neg_log_prob.mean(), cause, x_enc, x_aug

    def test(self, x, x_time):
        n_sample, n_node, t, d = x.shape

        x_enc = x.permute(0,2,1,3).squeeze() # x: (n_sample, t, n)
        x_aug = causal_interve(x_enc, self.interve_level) # causal intervention

        ##### local multi-periods feature capture ######
        # embedding
        enc_emb = self.embedding(x_enc, x_time)
        aug_emb = self.embedding(x_aug, x_time)

        # feature capture
        enc_pyramid, enc_weight = self.feat_ext(enc_emb)
        aug_pyramid, aug_weight = self.feat_ext(aug_emb)

        enc_out = self.layer_norm(self.feat_fusion(enc_pyramid, enc_weight))
        aug_out = self.layer_norm(self.feat_fusion(aug_pyramid, aug_weight))
        enc_out_proj = self.proj(enc_out + aug_out)

        h = enc_out_proj.reshape(n_sample, t, n_node, -1).permute(0,2,1,3)
        for layer in self.ST_ext:
            h = layer(h) # update by spatial temporal message interaction

        if self.use_multidim:
            x_nf = x.squeeze().permute(0,2,1).reshape(-1, n_node)
            h_nf = self.proj4nf(h.permute(0,2,3,1)).squeeze().reshape(-1, h.shape[3])
        else:
            x_nf = x.reshape((-1,d))
            h_nf = h.reshape((-1,h.shape[3]))

        log_prob = self.period_nf.log_prob(x_nf, h_nf).reshape([n_sample,-1])
        log_prob = log_prob.mean(dim=1)

        return -log_prob, h.permute(0,2,1,3).reshape(n_sample*n_node, -1, t), enc_out, aug_out

