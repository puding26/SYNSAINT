from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class SAINTSYNDataset(Dataset):   
    def __init__(self, samples, max_seq):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.data = []
        for problem_id in self.samples.index:
            time, elapsed, correct, ha, skillc= self.samples[problem_id]
            
            #if len(ques_id) > max_seq:
                #for l in range((len(ques_id)+max_seq-1)//max_seq):
                    #self.data.append(
                        #(time[l:l+max_seq], elapsed[l:l+max_seq], skillc[l:l+max_seq], correct[l:l+max_seq], ha[l:l+max_seq]))
            #else:
        self.data.append((time, elapsed, correct, ha, skillc))
            
            
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        time_, elapsed_, correct_, ha_,  skillc_= self.data[index]
        seq_len = len(skillc_)

        
        timezero = np.zeros(self.max_seq, dtype=int)
        elapsedzero= np.zeros(self.max_seq, dtype=int)
        skillczero = np.zeros(self.max_seq, dtype=int)
        correctzero = np.zeros(self.max_seq, dtype=int)
        #correctnesszero = np.zeros(self.max_seq, dtype=float)
        hazero = np.zeros(self.max_seq, dtype=float)
        
        if seq_len < self.max_seq:
            timezero[-seq_len:] = time_
            elapsedzero[-seq_len:] = elapsed_
            skillczero[-seq_len:] = skillc_
            correctzero[-seq_len:] = correct_
            #correctnesszero[-seq_len:] = correctness_
            hazero[-seq_len:] = ha_
        else:
            timezero[:] = time_[-self.max_seq:]
            elapsedzero[:] = elapsed_[-self.max_seq:]
            skillczero[:] = skillc_[-self.max_seq:]
            correctzero[:] = correct_[-self.max_seq:]
            #correctnesszero[:] = correctness_[-self.max_seq:]
            hazero[:] = ha_[-self.max_seq:]
    
        
        input_rtime = np.zeros(self.max_seq, dtype=int)
        input_rtime = np.insert(elapsedzero, 0, 0)
        input_rtime = np.delete(input_rtime, -1)
        
        input = {"input_rtime": input_rtime.astype(
            int), "hardall": hazero, "newskill": skillczero}

        return input, correctzero    


def get_dataloaders():
    dtypes = {'index': 'int64', 'problemId': 'int64', 'startTime': 'int64', 'timeTaken': 'float64', 
              'newskill':  'int64', 'problemId': 'float64', 'correct': 'int64', 'AveCorrect': 'float64', 'hardall': 'float64'}
    print("loading csv.....")
    train_df = pd.read_csv(Config.TRAIN_FILE, usecols=[
                           2, 3, 4, 6, 10, 11], dtype=dtypes)
    print("shape of dataframe :", train_df.shape)
    print(train_df.head())
  
    train_df = train_df.sort_values(by='startTime', ascending=True).reset_index(drop=True)
    n_skills = train_df['newskill'].nunique()
    print("no. of skills :", n_skills)
    print(train_df.isnull().sum())

    
    # grouping based on question_id
    print("Grouping questions...")
    group = train_df[["problemId", "startTime", "timeTaken", "correct", "hardall", "newskill"]]\
        .groupby("problemId")\
        .apply(lambda r: (r.startTime.values, r.timeTaken.values,
                          r.correct.values, r.hardall.values, r.newskill.values))
    #del train_df

    #gc.collect()
    print("splitting")
    train_df_features=train_df.drop(columns='correct')
    train_df_label=train_df['correct']
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for train_index,valid_index in sss.split(train_df_features, train_df_label):
        X_train=train_df_features.iloc[train_index]
        y_train=train_df_label.iloc[train_index]
        X_test=train_df_features.iloc[valid_index]
        y_test=train_df_label.iloc[valid_index]
        
    train = pd.concat([X_train,y_train],axis=1)
    val = pd.concat([X_test,y_test],axis=1)
    
    trainnew = train[["problemId", "startTime", "timeTaken", "correct", "hardall", "newskill"]]\
      .groupby("problemId")\
       .apply(lambda r: (r.startTime.values, r.timeTaken.values,
                          r.correct.values, r.hardall.values, r.newskill.values))
    valnew = val[["problemId", "startTime", "timeTaken", "correct", "hardall", "newskill"]]\
        .groupby("problemId")\
        .apply(lambda r: (r.startTime.values, r.timeTaken.values,
                         r.correct.values, r.hardall.values, r.newskill.values))                      
    #del train_df
    #train, val = train_test_split(group, test_size=0.2, random_state=42)
    print("train size: ", trainnew.shape, "validation size: ", valnew.shape)
    
    del train_df
    gc.collect()
    train_dataset = SAINTSYNDataset(trainnew, max_seq=Config.MAX_SEQ)
    
    val_dataset = SAINTSYNDataset(valnew, max_seq=Config.MAX_SEQ)
    # prepare for train dataset and valid dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=Config.BATCH_SIZE,
                            num_workers=8,
                            shuffle=False)
    del train_dataset, val_dataset
    gc.collect()
    return train_loader, val_loader
