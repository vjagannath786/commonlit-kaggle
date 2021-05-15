import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig
import numpy as np
import random



import config
import engine
from dataset import LitDataset, RobertaLitDataset
from model import LitModel, LitRoberta, LitRobertasequence




def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_predict_roberta(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    valid_fold = df.query(f"kfold == 4")
    #valid_fold = df

    targets = valid_fold['target']

    validset = RobertaLitDataset(valid_fold['excerpt'].values, targets=targets.values,is_test=False)

    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS,
    worker_init_fn=seed_worker)

    model_config = RobertaConfig.from_pretrained('roberta-base')
    model_config.output_hidden_states = True
    model_config.max_position_embeddings=514
    model_config.vocab_size = 50265
    model_config.type_vocab_size = 1
    
    model = LitRoberta(config= model_config)
    model.load_state_dict(torch.load(f'../../working/checkpoint_roberta_{fold}.pt'))

    model.to(config.DEVICE)

    valid_preds, valid_loss = engine.eval_fn(model, validloader)


    

    print(f"loss is {valid_loss}")
    return valid_loss



def run_predict_robertasequence(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    valid_fold = df.query(f"kfold == 4")
    #valid_fold = df

    targets = valid_fold['target']

    validset = RobertaLitDataset(valid_fold['excerpt'].values, targets=targets.values,is_test=False)

    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)

    model_config = RobertaConfig.from_pretrained('roberta-base')
    model_config.output_hidden_states = True    
    model_config.max_position_embeddings=514
    model_config.vocab_size = 50265
    model_config.type_vocab_size = 1
    
    model = LitRobertasequence(config= model_config)
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
    '''
    a = run_predict_roberta(0)
    b = run_predict_roberta(1)
    c = run_predict_roberta(2)
    d = run_predict_roberta(3)
    '''
    b = run_predict_roberta(3)

    #print(f'avg loss from both models {(a +b +c + d) / 4}')
    
