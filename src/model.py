import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
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

_layers = ['layer.23','layer.22','layer.21','layer.20','layer.19']


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
    def __init__(self,config, dropout):
        super(LitRoberta, self).__init__()
        self.roberta = AutoModel.from_pretrained('../../input/pretraining-large-external',  config=config)
        
        self.drop1 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(1024*2)
        self.l1 = nn.Linear(1024*2,1)
        
        '''
        self.drop1 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        self.l1 = nn.Linear(768*2,1)
        self.conv1 = nn.Conv1d(768*2,1,1)
        #self.batchnorm1 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.l2 = nn.Linear(128,64)
        self.drop3 = nn.Dropout(0.1)
        self.l3 = nn.Linear(64,1)
        '''
        #torch.nn.init.normal_(self.roberta.layer.11.weight, std =0.02)
        
        #self._init_weights(self.layer_norm)
        #self._init_weights(self.roberta.)
        #self._init_weights(self.roberta.layer.10)
        #self._init_weights(self.roberta.layer.9)
        #self._init_weights(self.roberta.layer.8)
        #self._init_weights(self.roberta.layer.7)
        #self.roberta.init_weights()

        self._init_weights(self.l1)
        self.weights_init_custom()
        
        #print(self.roberta.named_parameters())
        
        '''
        for params in self.roberta.named_parameters():
            name, weights = params
            #print(name)
        
            if any(i in name for i in _layers):
                #print('layer 11 freezed')
                weights.requires_grad = False
        '''



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            print('in init')
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def weights_init_custom(self):
        init_layers = [23, 22, 21,20,19]
        dense_names = ["query", "key", "value", "dense"]
        layernorm_names = ["LayerNorm"]
        for name, module in self.roberta.named_parameters():
            if any(f".{i}." in name for i in init_layers):
                if any(n in name for n in dense_names):
                    if "bias" in name:
                        module.data.zero_()
                    elif "weight" in name:
                        print(name)
                        module.data.normal_(mean=0.0, std=0.02)
                elif any(n in name for n in layernorm_names):
                    if "bias" in name:
                        module.data.zero_()
                    elif "weight" in name:
                        module.data.fill_(1.0)


    def forward(self,ids, mask, token_type_ids,targets=None):
        _out = self.roberta(ids, attention_mask = mask, token_type_ids= token_type_ids)
        x = _out['hidden_states']
        x = torch.cat((x[-1], x[-2]), dim=-1)
        #x = x[-1]
        
        #x = self.layer_norm(x)
        
        x = torch.mean(x,1, True)
        x = self.layer_norm(x)
        x = self.drop1(x)       
        #x = x.permute(0,2,1)
        #x = self.conv1(x)
        x = self.l1(x)
        #print(x.size())


        outputs =x.squeeze(-1)

        
        


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



class LitRNNRoberta(nn.Module):
    def __init__(self, config, dropout, n_layers, bidirectional):
        super(LitRNNRoberta, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base', config=config)

        self.rnn = nn.GRU(768,
                          256,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(256 * 2 if bidirectional else 256, 1)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,ids, mask, token_type_ids,targets=None):
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.roberta(ids, attention_mask = mask, token_type_ids=token_type_ids)[0]
                
        
        
        _, hidden = self.rnn(embedded)
        
        
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
               
        outputs = self.out(hidden)


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

     

