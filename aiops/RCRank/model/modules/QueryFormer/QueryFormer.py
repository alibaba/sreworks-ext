import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)


    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = hid

        return out

        
class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, tables = 1500, types=1500, joins = 1500, columns= 3000, \
                 ops=4, use_sample = True, use_hist = True, bin_number = 50):
        super(FeatureEmbed, self).__init__()
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)
        
        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size//8)

        self.linearFilter2 = nn.Linear(embed_size+embed_size//8, embed_size+embed_size//8)
        self.linearFilter = nn.Linear(embed_size+embed_size//8, embed_size+embed_size//8)

        self.linearType = nn.Linear(embed_size, embed_size)
        
        self.linearJoin = nn.Linear(embed_size, embed_size)
        
        self.linearSample = nn.Linear(1000, embed_size)
        
        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(joins, embed_size)
        
        use_hist = False
        self.use_hist = False
        if use_hist:
            self.project = nn.Linear(embed_size*5 + embed_size//8+1 + 4, embed_size*5 + embed_size//8+1 + 4)
        else:
            self.project = nn.Linear(embed_size*4 + embed_size//8 + 4, embed_size*4 + embed_size//8 + 4)
    
    def forward(self, feature):

        typeId, joinId, filtersId, filtersMask, table_sample, cost = torch.split(feature,(1,1,40,20,1001, 4), dim = -1)
        
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        
        tableEmb = self.getTable(table_sample)

        histEmb = None
        if self.use_hist:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb, cost), dim = 1)
        else:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, cost), dim = 1)
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())

        return emb.squeeze(1)
    
    def getTable(self, table_sample):
        table, sample = torch.split(table_sample,(1,1000), dim = -1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        
        if self.use_sample:
            emb += self.linearSample(sample)
        return emb
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())

        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        histExpand = hists.view(-1,self.bin_number,3).transpose(1,2)
        
        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0. 
        
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(emb, dim = 1)
        avg = total / num_filters.view(-1,1)
        
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        filterExpand = filtersId.view(-1,2,20).transpose(1,2)
        colsId = filterExpand[:,:,0].long()
        opsId = filterExpand[:,:,1].long()
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        
        concat = torch.cat((col, op), dim = -1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        
        concat[~filtersMask.bool()] = 0.
        
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(concat, dim = 1)
        avg = total / num_filters.view(-1,1)
                
        return avg
    


class QueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256, input_size = 1067
                ):
        
        super(QueryFormer,self).__init__()
        use_hist = False
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1 + 4
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 4
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        self.input_size = input_size

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim, pred_hid)
        self.pred_ln = nn.LayerNorm(pred_hid)

        self.pred2 = Prediction(hidden_dim, pred_hid)
        
    def forward(self, batched_data):
        attn_bias, rel_pos, x = batched_data["attn_bias"], batched_data["rel_pos"], batched_data["x"]
        
        heights = batched_data["heights"]     
        
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) 
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, self.input_size)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        output = self.pred(output)
        output = self.pred_ln(output)
        
        return output


class QueryFormerBert(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256, input_size = 1067
                ):
        
        super(QueryFormerBert,self).__init__()
        use_hist = False
        hidden_dim = 768
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        self.input_size = input_size

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim, pred_hid)
        self.pred_ln = nn.LayerNorm(pred_hid)

        self.pred2 = Prediction(hidden_dim, pred_hid)
        
    def forward(self, batched_data):
        attn_bias, rel_pos, x = batched_data["attn_bias"], batched_data["rel_pos"], batched_data["x"]
        
        heights = batched_data["heights"]     
        
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) 
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        node_feature = x
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        output = self.pred(output)
        output = self.pred_ln(output)
        
        return output






class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  
        v = v.transpose(1, 2)                  
        k = k.transpose(1, 2).transpose(2, 3)  

        q = q * self.scale
        x = torch.matmul(q, k)  
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  

        x = x.transpose(1, 2).contiguous()  
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


