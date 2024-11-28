import json
import torch.optim as optim
import sys
sys.path.append("../..")
from model.modules.LogModel.log_model import LogModel
import torch.nn as nn
import torch
from model.modules.QueryFormer.utils import *
from model.modules.QueryFormer.QueryFormer import QueryFormer
from transformers import BertTokenizer, BertModel
import pickle

class Predict(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, ff_dim, dropout=0.1):
        super(Predict, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.input_fc = nn.Linear(input_dim,1100,bias=False)  
        self.input_fc1 = nn.Linear(1100,model_dim,bias=False)  
    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_fc1(x)
        x = self.transformer_encoder(x)
        return x
    
class Alignment(nn.Module):
    def __init__(self,device):
        super(Alignment, self).__init__()
        self.flatten = nn.Flatten()
        self.plan_model = QueryFormer(pred_hid=32)
        self.sql_model = BertModel.from_pretrained("./bert-base-uncased")
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
        self.log_model = LogModel(input_dim=13, hidden_dim = 64, output_dim = 32)

        self.concat_dim_mask_plan = 17600
        self.predict_mask_plan = Predict(input_dim=self.concat_dim_mask_plan, model_dim=1024, num_heads=8, ff_dim=2048)
        self.Linear_mask_plan = nn.Linear(1024, 1067)

        self.concat_dim_mask_sql = 17600
        self.predict_mask_sql= Predict(input_dim=self.concat_dim_mask_sql, model_dim=768, num_heads=8, ff_dim=2048)
        self.Linear_mask_sql= nn.Linear(768, 768)

        self.device = device
    def forward(self, plan,sql, log,dic,mod):
        plan = self.plan_model(plan)
        sql = self.tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        sql = self.sql_model(**sql).last_hidden_state
        sql = self.transformer_encoder(sql).mean(dim=1)
        log = self.log_model(log)
        plan = self.flatten(plan)
        concatenated_vector = torch.cat((plan, sql, log, dic), dim=1).unsqueeze(1)
        if mod == 'mask_plan':
          transformer_output = self.predict_mask_plan(concatenated_vector)  
          transformer_output = transformer_output[:, 0, :]
          predicted_vector = self.Linear_mask_plan(transformer_output)
        elif mod == 'mask_sql':
          transformer_output = self.predict_mask_sql(concatenated_vector)
          transformer_output = transformer_output[:, 0, :]
          predicted_vector = self.Linear_mask_sql(transformer_output)
        pass
        return predicted_vector
      
  
if __name__ == "__main__":
    
    f1=open('mask_table_plan/encoding_1w.pickle','rb')
    encoding_mask_plan=json.dumps(pickle.load(f1).idx2table)
    f2=open('mask_table_sql/encoding1w.pickle','rb')
    encoding_mask_sql=json.dumps(pickle.load(f2).idx2table)
   
    device = 'cuda:2'
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    bert = BertModel.from_pretrained("./bert-base-uncased").to(device)    
    encoding_mask_plan = tokenizer(encoding_mask_plan, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    encoding_mask_sql = tokenizer(encoding_mask_sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)  
    encoding={'mask_plan':encoding_mask_plan,'mask_sql':encoding_mask_sql}
    model = Alignment(device).to(device)
    model.load_state_dict(torch.load('./model6.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs=500
    saved = 0
    best=0.0
    best_val_loss = float('inf') 
    batch_size=18
    batch_loss={'mask_sql':0.0,'mask_plan':0.0}
    for epoch in range(epochs):
       
        if epoch <= 6:
          print(epoch)
        else:
          i=0
          size=0
          model.train()  
          epoch_loss = 0.0
          file = {}
          file['mask_plan'] = open('mask_table_plan/plan_mask_1w.txt','r')
          file['mask_sql'] = open('mask_table_sql/sql_mask1w.txt','r')
          batch_list={}
          while i <= 10110:
            for mod in ['mask_sql','mask_plan']:
              for _ in range(batch_size):
                  line = file[mod].readline()
                  line=line.strip()
                  if not line:  
                      break
                  (query,x1,attn_bias,rel_pos,height,log_all,predict) = json.loads(line)
                  size=size+1
                  if len(batch_list.keys()) == 0:
                      batch_list['query']=[query]
                      plan={}
                      plan['x']= torch.tensor(x1).to(torch.float32).to(device) 
                      plan["attn_bias"] = torch.tensor(attn_bias).to(device) 
                      plan["rel_pos"] = torch.tensor(rel_pos).to(device)
                      plan["heights"] = torch.tensor(height).to(device) 
                      batch_list['plan']=plan
                      batch_list['log'] = torch.tensor(log_all).unsqueeze(0).to(device) 
                      batch_list['predict'] =  torch.tensor(predict).to(device).unsqueeze(0)
                  else:
                      batch_list['query'].append(query)
                      batch_list['plan']['x'] = torch.cat((batch_list['plan']['x'],torch.tensor(x1).to(torch.float32).to(device)),dim=0)
                      batch_list['plan']["attn_bias"]= torch.cat([batch_list['plan']['attn_bias'],torch.tensor(attn_bias).to(device)],dim=0) 
                      batch_list['plan']["rel_pos"]= torch.cat([batch_list['plan']['rel_pos'],torch.tensor(rel_pos).to(device)],dim=0)
                      batch_list['plan']["heights"]= torch.cat([batch_list['plan']['heights'],torch.tensor(height).to(device)],dim=0 )
                      batch_list['log'] =torch.cat([batch_list['log'],torch.tensor(log_all).unsqueeze(0).to(device)],dim=0 )
                      batch_list['predict'] =torch.cat([batch_list['predict'], torch.tensor(predict).unsqueeze(0).to(device)],dim=0)
              
              optimizer.zero_grad()
              sql, plan, log,predict = batch_list["query"], batch_list["plan"],batch_list["log"],batch_list['predict']
              dic = bert(**encoding[mod]).pooler_output
              dic = dic.repeat(len(sql), 1)
              output = model(plan,sql,log,dic,mod)
              loss = criterion(output, predict)
              loss.backward()
              optimizer.step()
              batch_loss[mod] = loss.item()
              batch_list={}
              if mod == 'mask_sql':
                i=i+batch_size
          


          if  (epoch+1) %3 == 0:
              torch.save(model.state_dict(), f'model{(epoch+1)}.pth')

         
