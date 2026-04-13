#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
from zipfile import ZipFile
import pandas as pd
import numpy as np
import pandas as pd
import glob
import re
import sklearn
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
import torch
from torch import nn
import time


# In[2]:


new=pd.read_csv('./KT1/merge.csv')
new.head()


# In[3]:


questions=pd.read_csv('./questions.csv')


# In[4]:


print(questions)


# In[5]:


import time
start = time.time()


# In[6]:


newdemo=new.set_index('question_id')


# In[7]:


questionsdemo=questions.set_index('question_id')


# In[8]:


mergenew=newdemo.merge(questionsdemo['skillcluster'],on='question_id')


# In[9]:


mergenew['elapsed_time']=pd.to_numeric(mergenew['elapsed_time'],errors='coerce')


# In[10]:


mergenew['elapsed_time'].quantile([0.25,0.5,0.75]


# In[11]:


print(mergenew)


# In[12]:


#mergenew=mergenew.set_index('question_id')
mergenew=mergenew.merge(questionsdemo['correct_answer'],on='question_id')


# In[13]:


mergenew['question_id']=mergenew.index
mergenew['correct']=(mergenew['user_answer']==mergenew['correct_answer']).apply(int)
print(mergenew.head())


# In[14]:


conditions= [
    (mergenew['elapsed_time']<=16000 ),
    (mergenew['elapsed_time'] > 16000) & (mergenew['elapsed_time']<=21000 ),
    (mergenew['elapsed_time'] > 21000) & (mergenew['elapsed_time'] <=30000 ),
    (mergenew['elapsed_time'] > 30000)
    ]
values = [0, 1, 2, 3]
mergenew['hard1'] = np.select(conditions, values)
print(mergenew.head())


# In[15]:


mergenew=mergenew.reset_index(drop=True)


# In[16]:


question_agg = mergenew.groupby('question_id')['correct'].agg(['sum', 'count'])

# obtain correctness
mergenew['correctness'] = mergenew['question_id'].map(question_agg['sum'] / question_agg['count'])
conditions2=[
    (mergenew['correctness']<=0.3),
    (mergenew['correctness']>0.3) & (mergenew['correct']<=0.5),
    (mergenew['correctness']<=0.5) & (mergenew['correct']<=0.7),
    (mergenew['correctness']>0.7)
]
values2=[3,2,1,0]


# In[17]:


mergenew['hard2']=np.select(conditions2,values2)


# In[18]:


batch_size = mergenew.shape[0]
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
loss = loss * sample_weight
lossnew=1-loss


# In[19]:


mergenew['hardall']=mergenew['hard1']*loss.detach().numpy()+mergenew['hard2']*lossnew.detach().numpy()
#print(mergenew.head())
end=time.time()

all_time=end-start
print('all training time is', all_time)


# In[ ]:




