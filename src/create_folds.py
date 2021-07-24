from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import os
import numpy as np


path = '../../input'

def create_folds(data, num_splits):
    

    data['kfold'] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)



    _bins = int(np.floor(1 + np.log2(len(data))))

    data['bins'] = pd.cut(data['target'], bins=_bins, labels=False)

    skf = KFold(n_splits=num_splits)

    for f, (t_,v_) in enumerate(skf.split(X=data, y=data.bins.values)):
        data.loc[v_,"kfold"] = f
    
    data = data.drop('bins', axis=1)

    return data

    




if __name__ == "__main__":

    df = pd.read_csv(os.path.join(path,'train.csv'))

    data = create_folds(df, 5)

    data.to_csv(os.path.join(path,'train_kfolds.csv'), index=False)

    print(data['kfold'].value_counts())