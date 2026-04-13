#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd




# In[2]:


import numpy as np
import pandas as pd
import glob
import time



# In[15]:

start_time=time.time()

questions=pd.read_csv('./assist2017.csv')


# In[25]:


questions.dropna(axis=0,how='any',inplace=True)


# In[29]:


questions['timeTaken'].quantile([0.25,0.5,0.75])


# In[30]:


conditions= [
    (questions['timeTaken']<=5),
    (questions['timeTaken'] > 5) & (questions['timeTaken']<=13),
    (questions['timeTaken'] > 13) & (questions['timeTaken'] <=34),
    (questions['timeTaken'] > 34)
    ]
values = [0, 1, 2, 3]
questions['hard1'] = np.select(conditions, values)
print(questions.head())


# In[31]:


conditions2=[
    (questions['AveCorrect']<=0.3),
    (questions['AveCorrect']>0.3) & (questions['AveCorrect']<=0.5),
    (questions['AveCorrect']<=0.5) & (questions['AveCorrect']<=0.7),
    (questions['AveCorrect']>0.7)
]
values2=[3,2,1,0]

questions['hard2']=np.select(conditions2,values2)
print(questions.head())


# In[32]:


import torch
from torch import nn
batch_size = questions.shape[0]
nb_classes = 2

model = nn.Linear(nb_classes, nb_classes)
weight = torch.empty(nb_classes).uniform_(0, 1)

criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

# This would be returned from your DataLoader
x = torch.randn(batch_size, nb_classes)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
sample_weight = torch.empty(batch_size).uniform_(0, 1)

output = model(x)
loss = criterion(output, target)
# times weights，derive means
loss = loss * sample_weight
lossnew=1-loss
print(loss)
print(lossnew)


# In[33]:


questions['hardall']=questions['hard1']*loss.detach().numpy()+questions['hard2']*lossnew.detach().numpy()
print(questions.head())


# In[34]:


questionsnew=questions[['studentId','problemId','timeTaken','startTime','AveCorrect','correct','hard1','hard2','hardall']]
questionsnew.head()
end_time=time.time()
all_time=end_time-start_time
print('all time is', all_time)

