from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AdamW,AdamWeightDecay
from transformers import (get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup)
from transformers import AutoConfig
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.metrics import mean_squared_error

from torch.utils.data import (
    Dataset, DataLoader, 
    SequentialSampler, RandomSampler
)

random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

import config
import engine_early
from dataset import LitDataset, RobertaLitDataset
from model import LitModel, LitRoberta, LitRobertasequence, LitRNNRoberta
from pytorrchtools import EarlyStopping


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def _loss_fn(targets, outputs):
    #print(outputs.shape)
    #print(targets.unsqueeze(1).shape)
    
    #loss = 
    #print(loss.view(-1))
    #print((outputs))
    #print((targets))
    return mean_squared_error(targets, outputs, squared=False)


bert_pred = None
roberta_pred = None



def get_optimizer_params(model):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = 2e-5
    no_decay = ['bias', 'gamma', 'beta']
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr':2e-5, "momentum" : 0.99},
    ]
    return optimizer_parameters


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk0:
            #a
            for key, value in data.items():
                #a
                data[key] = value.to(config.DEVICE)
            batch_preds, loss = model(**data)
            #print(batch_preds)
            #batch_preds = tuple(pred.cpu().detach().numpy() for pred in batch_preds)
            fin_preds.append(batch_preds.cpu().detach().numpy())
            fin_loss += loss.item()
            

    #print(f'------------{len(fin_preds)}-----------')
    #print(fin_preds)
    return fin_preds, fin_loss / len(data_loader)

def load_classifier(bert_outputs, roberta_outputs,valid_targtes):
    pass






def run_training(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    #df = df.query(f"kfold != 4")
    '''

    fold1 = df.query(f"kfold == {fold}")
    fold2 = df.query(f"kfold == {fold+1}")
    fold3 = df.query(f"kfold == {fold+2}")
    '''

    #train_fold = fold1.append(fold2)
    train_fold = df[df.kfold != fold]
    print(train_fold.shape)

    valid_fold = df.query(f"kfold == {fold}")
    print(valid_fold.shape)

    trainset = LitDataset(train_fold['excerpt'].values, targets= train_fold['target'].values, is_test=False)
    validset = LitDataset(valid_fold['excerpt'].values, targets= valid_fold['target'].values, is_test=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = config.TRAIN_BATCH_SIZE, num_workers = config.NUM_WORKERS)
    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)


    model = LitModel()
    
    model.to(config.DEVICE)

    parameter_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in parameter_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in parameter_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_fold) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    #optimizer = torch.optim.Adam(optimizer_parameters, lr=3e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau()

    early_stopping = EarlyStopping(patience=3, path=f'../../working/checkpoint_{fold}.pt',verbose=True)

    best_loss = 1000
    for epoch in range(config.EPOCHS):
        print(f'epoch {epoch} and lr is {scheduler.get_last_lr()}')
        train_loss = engine_early.train_fn(model, trainloader, optimizer, scheduler)
        valid_preds, valid_loss = engine_early.eval_fn(model, validloader)

        


        

        print(f'train_loss {train_loss} and valid_loss {valid_loss}')
        '''

        if valid_loss < best_loss:
            print('valid loss improved saving model')
            torch.save(model.state_dict(), f'model_{fold}.bin')
            best_loss = valid_loss
        '''

        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            bert_pred = valid_preds


        #model.load_state_dict(torch.load(f'checkpoint_{fold}.pt'))

        #return valid_preds







def run_roberta_training(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    
    #df = df.query(f"kfold != 4")

    #fold1 = df.query(f"kfold == {fold}")
    #fold2 = df.query(f"kfold == {fold+1}")
    #fold3 = df.query(f"kfold == {fold+2}")

    #shuffle data to prevent overfitting
    #df = df.sample(frac=1).reset_index(drop=True)

    

    #train_fold = fold1.append(fold2)
    #train_fold = train_fold.append(fold3)

    train_fold = df[df.kfold != fold]

    valid_fold = df.query(f"kfold == {fold}")

    trainset = RobertaLitDataset(train_fold['excerpt'].values, targets= train_fold['target'].values, is_test=False, max_lnth=config.MAX_LEN)
    validset = RobertaLitDataset(valid_fold['excerpt'].values, targets= valid_fold['target'].values, is_test=False, max_lnth = config.MAX_LEN)

    trainsampler = RandomSampler(trainset)
    validsampler = SequentialSampler(validset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = config.TRAIN_BATCH_SIZE, sampler=trainsampler,num_workers = config.NUM_WORKERS)
    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, sampler=validsampler,num_workers = config.NUM_WORKERS)

    model_config = AutoConfig.from_pretrained('/kaggle/input/robertaitpt')
    model_config.output_hidden_states = True
    #model_config.return_dict = False
    #model_config.max_position_embeddings=514
    #model_config.vocab_size = 50265
    #model_config.type_vocab_size = 1
    
    model = LitRoberta(config= model_config, dropout=0.3)
    model.to(config.DEVICE)

    '''
    for param in list(model.named_parameters()):
        name, _weights = param
        #print(name)        
        if 'pooler.dense.weight' in name:
            print('dense layer weight set to false')
            _weights.requires_grad = False
    '''

    
    parameter_optimizer = list(model.named_parameters())
    no_decay = ["bias","LayerNorm.weight","LayerNorm.bias"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in parameter_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-3,
        },
        {
            "params": [
                p for n, p in parameter_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    opt_param = get_optimizer_params(model)
    
    
    num_train_steps = (len(train_fold) //config.TRAIN_BATCH_SIZE * 13)
    print(num_train_steps)
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    #optimizer = AdamWeightDecay(learning_rate=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=num_train_steps)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    #optimizer = torch.optim.Adam(optimizer_parameters, lr=3e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau()

    early_stopping = EarlyStopping(patience=4, path=f'../../working/checkpoint_roberta_{fold}_v1.pt',verbose=True)

    best_loss = 1000
    #global_step = 10
    _higher = 20
    _lower = 20
    global_break = False
    for epoch in range(config.EPOCHS):
        

        print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
        #train_loss = engine_early.train_fn(model, trainloader, optimizer, scheduler,validloader, global_step,  early_stopping,roberta_pred, global_break, fold)
        #valid_preds, valid_loss = engine_early.eval_fn(model, validloader)

        model.train()
        fin_loss = 0
        tk0 = tqdm(trainloader, total=len(trainloader))

        for i,data in enumerate(tk0):
            
            
            
            

            i=i+1

            for key, value in data.items():

                data[key] = value.to(config.DEVICE)
        
        
            optimizer.zero_grad()
            _, loss = model(**data)       
        
        
            loss.backward()
            optimizer.step()

            
            
            
            
            if i % (_lower if i > 90 else _higher) == 0:
                valid_preds, valid_loss = eval_fn(model, validloader)
                print(valid_loss)
                early_stopping(valid_loss, model)
            
                if early_stopping.early_stop:

                    print("Early stopping")
                    global_break = True
                    print(global_break)
                    break
                
                if valid_loss < best_loss:
                    print(f'------------------ best loss is {valid_loss}----------')
                    roberta_pred = valid_preds
                    best_loss= valid_loss
            
            scheduler.step()
            fin_loss += loss.item()
            if i % (_lower if i > 90 else _higher) == 0:
                train_loss = fin_loss / i
                print(f'train_loss {train_loss} and valid_loss {valid_loss}')
            
        

        
        
        
        
        
        if global_break:
            print('breaking from epoch')
            break

        #scheduler.step()
        



        


        
        '''
        print(f'train_loss {train_loss} and valid_loss {valid_loss}')
        

        if valid_loss < best_loss:
            print('valid loss improved saving model')
            torch.save(model.state_dict(), f'model_{fold}.bin')
            best_loss = valid_loss
        
        #scheduler.step()

        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            roberta_pred = valid_preds



        #model.load_state_dict(torch.load(f'checkpoint_{fold}.pt'))
        '''


    return roberta_pred




def run_roberta_sequence_training(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    #df = df.query(f"kfold != 4")

    #fold1 = df.query(f"kfold == {fold}")
    #fold2 = df.query(f"kfold == {fold+1}")
    #fold3 = df.query(f"kfold == {fold+2}")

    #shuffle data to prevent overfitting
    #df = df.sample(frac=1).reset_index(drop=True)

    

    #train_fold = fold1.append(fold2)
    #train_fold = train_fold.append(fold3)

    train_fold = df[df.kfold != fold]

    valid_fold = df.query(f"kfold == {fold}")

    trainset = RobertaLitDataset(train_fold['excerpt'].values, targets= train_fold['target'].values, is_test=False,max_lnth=config.MAX_LEN)
    validset = RobertaLitDataset(valid_fold['excerpt'].values, targets= valid_fold['target'].values, is_test=False, max_lnth=config.MAX_LEN)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = config.TRAIN_BATCH_SIZE, num_workers = config.NUM_WORKERS)
    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)

    model_config = RobertaConfig.from_pretrained('roberta-base')
    model_config.output_hidden_states = True
    model_config.return_dict = True
    #model_config.max_position_embeddings=514
    #model_config.vocab_size = 50265
    #model_config.type_vocab_size = 1
    
    model = LitRobertasequence(config= model_config, dropout=0.1)
    model.to(config.DEVICE)

    parameter_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in parameter_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in parameter_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_fold) / config.TRAIN_BATCH_SIZE * 15)
    #num_train_steps = 3


    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    #optimizer = torch.optim.Adam(optimizer_parameters, lr=3e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau()

    early_stopping = EarlyStopping(patience=4, path=f'../../working/checkpoint_roberta_sequence_{fold}.pt',verbose=True)

    best_loss = 1000
    for epoch in range(config.EPOCHS):
        print(f'epoch {epoch} and lr is {scheduler.get_last_lr()}')
        train_loss = engine_early.train_fn(model, trainloader, optimizer, scheduler)
        valid_preds, valid_loss = engine_early.eval_fn(model, validloader)

        

        


        

        print(f'train_loss {train_loss} and valid_loss {valid_loss}')
        '''

        if valid_loss < best_loss:
            print('valid loss improved saving model')
            torch.save(model.state_dict(), f'model_{fold}.bin')
            best_loss = valid_loss
        '''

        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            roberta_pred = valid_preds



        #model.load_state_dict(torch.load(f'checkpoint_{fold}.pt'))

    return valid_preds    



    




if __name__ == "__main__":
    #run_training(3)
    
    _outputs = []
    _targets = []
    

    for i in range(1):
        df = pd.read_csv(config.TRAIN_FILE)
        
        tmp_target = df.query(f"kfold == {0}")['target'].values
        tmp = run_roberta_training(0)

        #print(tmp)

        #tmp = np.ravel(tmp)
        #tmp1 = np.ravel(tmp)

        #print(tmp1)
        

        

        a = np.concatenate(tmp,axis=0)
        b = np.concatenate(a, axis=0)

        #print(len(b))
        #print(len(tmp_target))

        loss =  _loss_fn(tmp_target, b)

        print(f'loss for fold {i} is {loss}')

        #print(b)
        #print(tmp_target)
        _outputs.append(b)
        _targets.append(tmp_target)

    

    c = np.concatenate(_outputs, axis=0)
    d = np.concatenate(_targets,axis=0)
    total_loss =  _loss_fn(d,c)

    print(f'total loss for for all folds is {total_loss}')

    




    #print(bert_pred)
    #print(roberta_pred)

    

    

    

    
