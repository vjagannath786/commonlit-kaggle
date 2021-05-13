import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.ensemble import RandomForestRegressor




import config
import engine
from dataset import LitDataset, RobertaLitDataset
from model import LitModel, LitRoberta
from pytorrchtools import EarlyStopping



def loss_fn(outputs, targets):
    #print(outputs.shape)
    #print(targets.unsqueeze(1).shape)
    
    #loss = 
    #print(loss.view(-1))
    return torch.sqrt(nn.MSELoss()(outputs, targets))


bert_pred = None
roberta_pred = None


def load_classifier(bert_outputs, roberta_outputs,valid_targtes):
    pass






def run_training(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    df = df.query(f"kfold != 4")
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

    early_stopping = EarlyStopping(patience=5, path=f'../../working/checkpoint_{fold}.pt',verbose=True)

    best_loss = 1000
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, trainloader, optimizer, scheduler)
        valid_preds, valid_loss = engine.eval_fn(model, validloader)

        


        

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

    fold1 = df.query(f"kfold == {fold}")
    fold2 = df.query(f"kfold == {fold+1}")
    fold3 = df.query(f"kfold == {fold+2}")

    

    train_fold = fold1.append(fold2)
    train_fold = train_fold.append(fold3)

    valid_fold = df.query(f"kfold == {fold+3}")

    trainset = RobertaLitDataset(train_fold['excerpt'].values, targets= train_fold['target'].values, is_test=False)
    validset = RobertaLitDataset(valid_fold['excerpt'].values, targets= valid_fold['target'].values, is_test=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = config.TRAIN_BATCH_SIZE, num_workers = config.NUM_WORKERS)
    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)


    
    model = LitRoberta()
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
    #num_train_steps = 3


    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    #optimizer = torch.optim.Adam(optimizer_parameters, lr=3e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau()

    early_stopping = EarlyStopping(patience=5, path=f'../../working/checkpoint_roberta_{fold}.pt',verbose=True)

    best_loss = 1000
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, trainloader, optimizer, scheduler)
        valid_preds, valid_loss = engine.eval_fn(model, validloader)

        


        

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

        #return valid_preds




    



    




if __name__ == "__main__":
    run_training(0)
    run_roberta_training(0)

    print(bert_pred)
    print(roberta_pred)

    

    

    

    