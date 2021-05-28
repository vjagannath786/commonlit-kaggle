from tqdm import tqdm
import torch
import config


def train_fn(model, data_loader, optimizer, scheduler, validloader, global_step,early_stopping,roberta_pred, global_break):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for i,data in enumerate(tk0):

        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        
        
        optimizer.zero_grad()
        _, loss = model(**data)
        
        
        
        loss.backward()
        optimizer.step()

        if i % global_step == 0:
            valid_preds, valid_loss = eval_fn(model, validloader)
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:

                print("Early stopping")
                global_break = True
                break
            else:
                roberta_pred = valid_preds
		
        scheduler.step()
        fin_loss += loss.item()
    train_loss = fin_loss / len(data_loader)
    print(f'train_loss {train_loss} and valid_loss {valid_loss}')
    return fin_loss / len(data_loader)


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
            fin_loss += loss.item()
            fin_preds.append(batch_preds.cpu().detach().numpy())

    
    return fin_preds, fin_loss / len(data_loader)
