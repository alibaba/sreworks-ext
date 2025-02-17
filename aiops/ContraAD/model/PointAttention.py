import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from .attend import Attend
from torch.nn import Module, ModuleList
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from typing import Optional, Union, Tuple

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from .RevIN import RevIN

from rotary_embedding_torch import RotaryEmbedding


def normalize(x,method='z-score'):
    # x shape : batch, size, channel
    # min-max normalization
    if len(x.shape) <3:
        x = x.unsqueeze(dim=-1)
    b,w,c = x.shape
    min_vals,_ = torch.min(x,dim=1) # batch,channel
    max_vals,_ = torch.max(x,dim=1) # batch,channel
    mean_vals = torch.mean(x,dim=1)
    std_vals = torch.std(x,dim=1)
    if method == 'min-max':
        min_vals = repeat(min_vals,'b c -> b w c',w=w)
        max_vals = repeat(max_vals,'b c -> b w c',w=w)
        x = (x - min_vals) / (max_vals - min_vals + 1e-8)
    # z-score normalization
    elif method == 'z-score':
        mean_vals = repeat(mean_vals,'b c -> b w c',w=w)
        std_vals = repeat(std_vals,'b c -> b w c',w=w)
        x = torch.abs((x - mean_vals) / (std_vals + 1e-8))
    # softmax normalization
    elif method == 'softmax':
        x = F.softmax(x,dim=1)
    else :
        raise ValueError('Unknown normalization method')
    if c ==1:
        x = x.squeeze(dim=-1)
    return x


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def identity(t, *args, **kwargs):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        heads=4,
        dropout=0.0,
        causal=False,
        flash=True,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, dim_inner, bias=False),
            nn.SiLU(),
            Rearrange("b n (h d) -> b h n d", h=heads),
        )

        self.attend = Attend(flash=flash, dropout=dropout, causal=causal)

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        out = self.attend(q, k, v)

        out = out * self.to_v_gates(x)
        return self.to_out(out)


# feedforward


class GEGLU(Module):
    def forward(self, x):
        x, gate = rearrange(x, "... (r d) -> r ... d", r=2)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim),
    )


# transformer block


class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim,
        causal=False,
        dim_head=32,
        heads=8,
        ff_mult=4,
        flash_attn=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb

        self.attn = Attention(
            flash=flash_attn,
            rotary_emb=rotary_emb,
            causal=causal,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
        )
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x, rotary_emb: Optional[RotaryEmbedding] = None):
        x = self.attn(x) + x
        x = self.attn_norm(x)

        x = self.ff(x) + x
        x = self.ff_norm(x)

        return x

def slide_window(x,hop=4):
    # b c w 
    x_pad = F.pad(x,(hop,hop),"replicate")
    return x_pad.unfold(2,hop*2+1,1)

class PatchAttention(nn.Module):
    def __init__(
        self,
        win_size:int,
        channel:int,
        depth:int=4,
        dim:int=512,
        dim_head:int=128,
        heads:int=4,
        attn_dropout:float=0.2,
        ff_mult:int=4,
        ff_dropout:float=0.2,
        hop:int=4,
        reduction='mean',
        use_RN=False,
        flash_attn=True,
    ):
        super().__init__()
        self.win_size = win_size
        self.channel = channel
        self.hop = hop
        
        self.intra_layer = nn.ModuleList([])
        self.reduction = reduction
        rotary_emb = RotaryEmbedding(dim_head)
        for _ in range(depth):
            self.intra_layer.append(
                ModuleList(
                    [
                        Attention(
                            dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            flash=flash_attn,
                            rotary_emb=rotary_emb
                        ),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
                        nn.LayerNorm(dim),
                    ]
                )
            )
        patch_size = 2 * hop +1 
        self.intra_revin = RevIN(patch_size)
        self.intra_in = nn.Sequential(
            # (b r ) p c 
            nn.Linear(channel,dim),
            nn.LayerNorm(dim)
        )
        self.intra_seq_revin = RevIN(win_size)
        # self.to_out = nn.Linear(win_size * dim,win_size)
   

    def forward(self,input_x):
        b,w,c = input_x.shape
        x = rearrange(input_x,'b w c -> b c w')
        x = slide_window(x,self.hop) # batch channel rolling_num patch_size
        x = rearrange(x,'b c r p -> (b r) p c')
        #TODO:// check revin dim 
        intra_x,reverse_fn = self.intra_revin(x)
        intra_x = self.intra_in(intra_x)
        for attn,attn_post_norm,ff,ff_post_norm in self.intra_layer:
            intra_x = attn(intra_x) + intra_x
            intra_x = attn_post_norm(intra_x)
            intra_x = ff(intra_x) + intra_x
            intra_x = ff_post_norm(intra_x)


        intra_x = rearrange(intra_x,'(b r) p d -> b r p d' ,b=b)
        intra_x = reduce(intra_x,'b r p d -> b r d',self.reduction)


        intra_x_seq = rearrange(input_x, "b w c -> b w c")
        intra_x_seq, reverse_fn = self.intra_seq_revin(intra_x_seq)
        intra_x_seq = self.intra_in(intra_x_seq)
        for attn, attn_post_norm, ff, ff_post_norm in self.intra_layer:
            intra_x_seq = attn(intra_x_seq) + intra_x_seq
            intra_x_seq = attn_post_norm(intra_x_seq)
            intra_x_seq = ff(intra_x_seq) + intra_x_seq
            intra_x_seq = ff_post_norm(intra_x_seq)
        return intra_x + intra_x_seq
        # x = rearrange(intra_x + intra_x_seq,'b w d -> b (w d)')
        # return self.to_out(x)


        

class FeatureDistance(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape (batch, num_variates, embedding_dim)
        dis = torch.cdist(x, x)
        return dis.sum(2).sum(1) / 2


def cal_metric(x,z_score,mode='z-score',soft=True,soft_mode='min',model_mode='train'):
    if mode =='z-score_mae':
        dis = torch.cdist(x,x).sum(2)
        if soft:
            if soft_mode=='sum':
                val = dis.sum(dim=1)
                val = repeat(val,"b -> b w", w=dis.size(1))
                dis = normalize(dis/val) # batch,win
            elif soft_mode == 'min':
                val,_ = dis.min(dim=1)
                val = repeat(val,"b -> b w" ,w=dis.size(1))
                dis = normalize(dis/val) # batch,win
        if model_mode =='train':
            return F.l1_loss(dis,z_score,reduction='mean'),dis
        else:
            return dis 
    elif mode == 'z_score_mse':
        dis = torch.cdist(x,x).sum(2)
        if soft:
            if soft_mode=='sum':
                val = dis.sum(dim=1)
                val = repeat(val,"b -> b w", w=dis.size(1))
                dis = normalize(dis/val) # batch,win
            elif soft_mode == 'min':
                val,_ = dis.min(dim=1)
                val = repeat(val,"b -> b w" ,w=dis.size(1))
                dis = normalize(dis/val) # batch,win
        if model_mode =='train':
            return F.mse_loss(dis,z_score,reduction='mean'),dis
        else:
            return dis 
    
    elif mode == 'z_score_clamp':
        dis = torch.cdist(x,x).sum(2)
        if soft:
            if soft_mode=='sum':
                val = dis.sum(dim=1)
                val = repeat(val,"b -> b w", w=dis.size(1))
                dis = normalize(dis/val) # batch,win
            elif soft_mode == 'min':
                val,_ = dis.min(dim=1)
                val = repeat(val,"b -> b w" ,w=dis.size(1))
                dis = normalize(dis/val) # batch,win
        if model_mode == 'train':
            return torch.where(dis>z_score,dis,z_score-dis).sum(dim=1).mean(),dis
        else:
            return dis 

    elif mode == 'distance':
        dis = torch.cdist(x,x).sum(2)
        if soft:
            if soft_mode=='sum':
                val = dis.sum(dim=1)
                val = repeat(val,"b -> b w", w=dis.size(1))
                dis = normalize(dis/val) # batch,win
            elif soft_mode == 'min':
                val,_ = dis.min(dim=1)
                val = repeat(val,"b -> b w" ,w=dis.size(1))
                dis = normalize(dis/val) # batch,win
        if model_mode == 'train':
            return dis.sum(dim=1).mean(),dis
        else:
            return dis 

class PointHingeLoss(Module):
    def __init__(self,mode='distance',soft=True,soft_mode='min'):
        super().__init__()
        self.mode = mode 
        self.soft = soft
        self.soft_mode = soft_mode

    def forward(self,x,z_score):
        loss,metric = cal_metric(x=x,z_score=z_score,mode=self.mode,soft=self.soft,soft_mode=self.soft_mode)
        return loss,metric



if __name__ == "__main__":
    x = torch.randn(2, 30, 3)
    # f = torch.randn(2,30,512)
    # z_score = torch.sum(normalize(x),dim=-1)
    # cri = PointHingeLoss()
    # loss = cri(f,z_score)
    # print(loss.shape)
    # PatchAttention()
    model = PatchAttention(30, 3, 4, 512, 128, 4, 0.2, 4, 0.2,4,'mean', False, True)
    intra_x = model(x)
    print(intra_x.shape)
