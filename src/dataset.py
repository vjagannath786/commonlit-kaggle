import config
import torch
import pandas as pd


class LitDataset:
    def __init__(self, review, targets=None, is_test= False):
        self.review = review
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.is_test = is_test

    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, idx):
        review = self.review[idx]
        

        inputs = self.tokenizer.encode_plus(review, None, truncation=True, add_special_tokens= True, max_length = config.MAX_LEN, 
        padding = 'max_length')

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.is_test:
            return {
                "ids": ids,
                "mask": mask,
                "token_type_ids": token_type_ids

            }
        else:
            targets = torch.tensor(self.targets[idx], dtype=torch.float)
            return {
                
                "ids": ids,
                "mask": mask,
                "token_type_ids": token_type_ids,
                "targets" : targets
                
            }





if __name__ == "__main__":
    df = pd.read_csv('..//input//train_folds.csv')

    dataset = LitDataset(review=df['excerpt'], targets=df['target'])

    print(dataset[1000])





