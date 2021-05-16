import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import config
from  dataset import LitDataset
import numpy as np
import pandas as pd





def loss_fn(outputs, targets):
    #print(outputs.shape)
    #print(targets.unsqueeze(1).shape)
    
    #loss = 
    #print(loss.view(-1))
    return torch.sqrt(nn.MSELoss()(outputs, targets))




class LitModel(nn.Module):
    def __init__(self):
        super(LitModel, self).__init__()
        self.model = transformers.BertModel.from_pretrained(config.BERT_MODEL, return_dict= False)
        self.drop = nn.Dropout(0.1)
        
        
        self.linear1 = nn.Linear(768,1)
        

        #self.linear2 = nn.Linear(128,64)
        

        
        #self.linear3 = nn.Linear(64,1)
        

    
    



    def forward(self, ids, mask, token_type_ids, targets=None):

        #print(ids)
        
        

        _, x = self.model(ids, attention_mask=mask, token_type_ids= token_type_ids)        
        #x = F.leaky_relu(x)
        x = self.drop(x)        
        x = self.linear1(x)
        
        #x = F.leaky_relu(x)
        #x = self.drop(x)
        
        #x= self.linear2(x)
        #x = F.leaky_relu(x)
        #x = self.drop(x)
        
        #x = self.linear3(x)
        outputs = x
        
        

        if targets is None:
            return outputs
        else:
            loss = loss_fn(outputs, targets.unsqueeze(1))
            #print(loss)
            return outputs, loss


class LitRoberta(nn.Module):
    def __init__(self,config, dropout=0.2):
        super(LitRoberta, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base',  config=config)
        
        self.drop1 = nn.Dropout(dropout)
        self.l1 = nn.Linear(768,1)
        #self.batchnorm1 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.l2 = nn.Linear(128,64)
        self.drop3 = nn.Dropout(0.1)
        self.l3 = nn.Linear(64,1)

        torch.nn.init.normal_(self.l1.weight, std =0.02)
    
    def forward(self,ids, mask, targets=None):
        x = self.roberta(ids, attention_mask = mask)
        x = x['hidden_states']
        x = x[-1]
        
        x = self.drop1(x)
        x = torch.mean(x,1, True)
        #print(x.size())
        x = self.l1(x)
        
        
        #x = self.drop2(x)
        #x = self.l2(x)
        #x = self.drop3(x)
        #x = self.l3(x)
        outputs = x.squeeze(-1)


        if targets is None:
            return outputs
        else:
            loss = loss_fn(outputs, targets.unsqueeze(1))
            return outputs, loss


class LitRobertasequence(nn.Module):
    def __init__(self,config, dropout):
        super(LitRobertasequence, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base',  config=config)
        
        self.drop1 = nn.Dropout(dropout)
        self.l1 = nn.Linear(768*1,1)
        #self.batchnorm1 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.l2 = nn.Linear(128,64)
        self.drop3 = nn.Dropout(0.1)
        self.l3 = nn.Linear(64,1)

        torch.nn.init.normal_(self.l1.weight, std =0.02)
    
    def forward(self,ids, mask, targets=None):
        x = self.roberta(ids, attention_mask = mask)
        x = x['hidden_states']
        x = x[-1]
        
        x = self.drop1(x)
        x = torch.mean(x,1, True)
        #print(x.size())
        x = self.l1(x)
        
        
        #x = self.drop2(x)
        #x = self.l2(x)
        #x = self.drop3(x)
        #x = self.l3(x)
        outputs = x.squeeze(-1)


        if targets is None:
            return outputs
        else:
            loss = loss_fn(outputs, targets.unsqueeze(1))
            return outputs, loss



if __name__ == "__main__":
     df = pd.read_csv('..//input//train_folds.csv')

     dataset = LitDataset(review=df['excerpt'], targets=df['target'])

     data = dataset[0]

     model = LitModel()
     outputs, loss = model(**data)

     

