from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from confignew import Config
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class SAINTSYNDataset(Dataset):   
    def __init__(self, samples, max_seq):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.data = []
        for ques_id in self.samples.index:
            time, elapsed, correct, ha, skillc= self.samples[ques_id]
            
            if len(ques_id) > max_seq:
                for l in range((len(ques_id)+max_seq-1)//max_seq):
                    self.data.append(
                        (time[l:l+max_seq], elapsed[l:l+max_seq], correct[l:l+max_seq], ha[l:l+max_seq], skillc[l:l+max_seq]))
            else:
                self.data.append((time, elapsed, correct, ha, skillc))
            
            
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        time_, elapsed_, correct_, ha_, skillc_= self.data[index]
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
    dtypes = {'timestamp': 'int64', 'elapsed_time': 'int32', 
              'newskill':  'int64', 'question_id': 'str', 'correct': 'int8', 'correctness': 'float64', 'hardall': 'float64'}
    print("loading csv.....")
    train_df = pd.read_csv(Config.TRAIN_FILE, usecols=[
                           0, 3, 6, 7, 11, 12], dtype=dtypes)
    print("shape of dataframe :", train_df.shape)
    print(train_df.head())
  
    train_df = train_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    n_skills = train_df['newskill'].nunique()
    print("no. of skills :", n_skills)
    print(train_df.isnull().sum())

    
    #train1=train_df[:10000]
    #color1 = np.arange(train1['skillcluster'][train1.correct==0].shape[0])
    #color2 = np.arange(train1['skillcluster'][train1.correct==1].shape[0])
    #fig = plt.figure(figsize=(20,10),dpi=600)
    #ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(train1['skillcluster'][train1.correct==0], train1['hardall'][train1.correct==0], train1['elapsed_time'][train1.correct==0], c='yellow', s=60, marker='o')
    #ax.scatter(train1['skillcluster'][train1.correct==1], train1['hardall'][train1.correct==1], train1['elapsed_time'][train1.correct==1], c='blue', s=60, marker='^')
    #ax.legend(['wrong answer','correct answer'],loc=1,fontsize=6)
    #ax.view_init(30,185)
    #from matplotlib import font_manager

    #ax.tick_params(labelsize=6) # font size for all axis

    #ax.set_xlabel('Skillcluster',fontsize=8,labelpad=2)
    #ax.set_ylabel('Hardness',fontsize=8,labelpad=2)
    #ax.set_zlabel('Time',fontsize=8,labelpad=2)
    #plt.xticks(position=(150,0,0))
    #ax.set_xticks(ticks=[0,25,50,75,100,125,150,175],labels=[0,25,50,75,100,125,150,175],position=(1000,1000,1000))
    #plt.show()
    
    # grouping based on question_id
    print("Grouping questions...")
    #group = train_df[["question_id", "timestamp", "elapsed_time", "skillcluster", "correct", "hardall"]]\
      #  .groupby("question_id")\
      #  .apply(lambda r: (r.timestamp.values, r.elapsed_time.values,
                       #   r.skillcluster.values, r.correct.values, r.hardall.values))
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
    #train, val = train_test_split(group, test_size=0.2, random_state=42)
    
    trainnew = train[["timestamp", "elapsed_time", "question_id", "correct", "hardall", "newskill"]]\
        .groupby("question_id")\
        .apply(lambda r: (r.timestamp.values, r.elapsed_time.values,
                          r.correct.values, r.hardall.values, r.newskill.values))
    
    valnew = val[["timestamp", "elapsed_time", "question_id", "correct", "hardall", "newskill"]]\
        .groupby("question_id")\
        .apply(lambda r: (r.timestamp.values, r.elapsed_time.values,
                          r.correct.values, r.hardall.values, r.newskill.values))
                         
    print("train size: ", trainnew.shape, "validation size: ", valnew.shape)
    
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
