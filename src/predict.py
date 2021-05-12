import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup



import config
import engine
from dataset import LitDataset
from model import LitModel




def run_predict(fold):
    df = pd.read_csv(config.TRAIN_FILE)
    #valid_fold = df.query(f"kfold == 4")
    valid_fold = df

    targets = valid_fold['target']

    validset = LitDataset(valid_fold['excerpt'].values, targets=targets.values,is_test=False)

    validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, num_workers = config.NUM_WORKERS)

    model = LitModel()
    model.load_state_dict(torch.load(f'checkpoint_{fold}.pt'))

    model.to(config.DEVICE)

    valid_preds, valid_loss = engine.eval_fn(model, validloader)


    

    print(f"loss is {valid_loss}")
















if __name__ == "__main__":
    run_predict(0)
    run_predict(1)
    run_predict(2)
    run_predict(3)

