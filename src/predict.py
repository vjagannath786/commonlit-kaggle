import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup



import config
import engine
from dataset import LitDataset, RobertaLitDataset
from model import LitModel, LitRoberta




def run_predict_roberta(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    valid_fold = df.query(f"kfold == 4")
    #valid_fold = df

    targets = valid_fold['target']

    validset = RobertaLitDataset(valid_fold['excerpt'].values, targets=targets.values,is_test=False)

    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)

    model = LitRoberta()
    model.load_state_dict(torch.load(f'../../working/checkpoint_roberta_{fold}.pt'))

    model.to(config.DEVICE)

    valid_preds, valid_loss = engine.eval_fn(model, validloader)


    

    print(f"loss is {valid_loss}")
    return valid_loss



def run_predict(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    valid_fold = df.query(f"kfold == 4")
    #valid_fold = df

    targets = valid_fold['target']

    validset = LitDataset(valid_fold['excerpt'].values, targets=targets.values,is_test=False)

    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)

    model = LitModel()
    model.load_state_dict(torch.load(f'../../working/checkpoint_{fold}.pt'))

    model.to(config.DEVICE)

    valid_preds, valid_loss = engine.eval_fn(model, validloader)


    

    print(f"loss is {valid_loss}")

    return valid_loss
















if __name__ == "__main__":
    a = run_predict_roberta(0)
    b = run_predict_roberta(1)
    #b = run_predict(0)

    #print(f'avg loss from both models {(a +b) / 2}')
    
