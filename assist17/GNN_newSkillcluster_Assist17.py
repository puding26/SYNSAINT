#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Begin data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as plt
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder


# In[3]:


nodes=pd.read_csv('./assist2017.csv')
print(nodes.head)


# In[4]:


node1=nodes[['skill','problemId','problemType', 'AveKnow', 'AveCarelessness', 'NumActions', 'AveResBored', 'AveResEngcon','AveResConf', 'AveResFrust', 'AveResOfftask', 'AveResGaming', 'assignmentId', 'assistmentId']]


# In[5]:


node_2 = node1.drop_duplicates()
print(node_2)


# In[6]:


node_2.drop_duplicates(subset=['problemId'], keep='first', inplace=True)


# In[7]:


node_2=node_2.dropna()
node_3=node_2.reset_index(drop=True)


# In[8]:


print(node_3)


# In[9]:


label_encoder = LabelEncoder()

node_3['skill'] = label_encoder.fit_transform(node_3['skill'])

node_3['problemType'] = label_encoder.fit_transform(node_3['problemType'])
newnode=node_3.drop(columns=['problemId','skill'])
y = node_3[['problemId','skill']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(newnode)
print(X_scaled)


# In[10]:


X=pd.DataFrame(X_scaled)
X.columns=['problemType', 'AveKnow', 'AveCarelessness', 'NumActions', 'AveResBored', 'AveResEngcon','AveResConf', 'AveResFrust', 'AveResOfftask', 'AveResGaming', 'assignmentId', 'assistmentId']
newnewnode=pd.concat([X,y],axis=1)
print(newnewnode)


# In[11]:


newnewnode.to_csv('assist17nodes.csv',index=None)


# In[12]:


#cosine-similarity
import pandas as pd
from itertools import permutations
from scipy.spatial.distance import cosine

# group by skills
result = []
for skill, group in newnewnode.groupby('skill'):
    # sole ids
    ids = group['problemId'].unique().tolist()
    
    
    pairs = list(permutations(ids, 2))
    #print(pairs)
    # calculate similarity
    for scr, d in pairs:
        scr_types = newnewnode.loc[newnewnode["problemId"] == scr].values.flatten()
        #scr_types1 = scr_types
        #print(scr_types)
        d_types = newnewnode.loc[newnewnode["problemId"] == d].values.flatten()
        #d_types1 = d_types.reshape(1,-1)
        #print(d_types)
        cos_distance = cosine(scr_types, d_types)
        similarity = 1 - cos_distance
        
        result.append({
            'Src': scr,
            'Dst': d,
            'weight': similarity
            #'skill': skill  # retain skill information
        })

# derive final DataFrame
new_df = pd.DataFrame(result)

# validate
print(new_df)


# In[13]:


ids=newnewnode['problemId'].values
ids_map = {k: i for i, k in enumerate(ids)}
print(ids_map)


# In[14]:


new_df['Src'] = new_df['Src'].map(ids_map).fillna(new_df['Src'])
new_df['Dst'] = new_df['Dst'].map(ids_map).fillna(new_df['Dst'])
print(new_df)


# In[15]:


new_df.to_csv('edgeinfo.csv',index=None)


# In[2]:


#Begin graph construction and training
new_df=pd.read_csv('edgeinfo.csv')
newnewnode=pd.read_csv('assist17nodes.csv')
ids=newnewnode['problemId'].values


# In[3]:


# Construction
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import time
start_time = time.time()
dgl_src = torch.from_numpy(new_df['Src'].to_numpy())
dgl_dst = torch.from_numpy(new_df['Dst'].to_numpy())
g = dgl.graph((dgl_src, dgl_dst), num_nodes=len(ids))
node_feature = torch.from_numpy(newnewnode.drop(columns=['problemId']).to_numpy())
g.ndata["feat"] = node_feature
g.edata['weight']= torch.tensor(new_df['weight'].values)

# 验证
#print("图结构:", g)
#print("特征矩阵:\n", g.ndata["feat"], g.edata['weight'])


# In[4]:


# Final GNN construction and training
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GNN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size).double()
        self.conv2 = dgl.nn.GraphConv(hidden_size, out_feats).double()
        
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.gelu(h)
        h = self.conv2(g, h)
        return F.normalize(h, p=2, dim=1)  # normalization

def contrastive_loss(embeddings, edge_src, edge_dst, edge_weights, temperature=0.5):
    # positive samples
    src_emb = embeddings[edge_src]
    dst_emb = embeddings[edge_dst]
    pred_similarity = F.cosine_similarity(src_emb, dst_emb, dim=1)  # [num_edges]
    
    # MSE loss
    mse_loss = F.mse_loss(pred_similarity, edge_weights)
    
    return mse_loss

def build_graph(num_nodes, edges, node_features, edge_weights):
    g = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
    g.ndata['feat'] = node_features
    g.edata['weight'] = edge_weights
    return g

def train(g, model, features, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        embeddings = model(g, features)
        edge_src, edge_dst = g.edges()
        loss = contrastive_loss(embeddings, edge_src, edge_dst, g.edata['weight'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            
# initiate
model = GNN(in_feats=13, hidden_size=16, out_feats=90) 
g = dgl.add_self_loop(g)

# training
train(g, model, node_feature)


# In[5]:


#derive final embedding
model.eval()
with torch.no_grad():
    embeddings = model(g, node_feature)
    #print("Learned embeddings shape:", embeddings.shape)


# In[6]:


from sklearn.cluster import HDBSCAN
hdb = HDBSCAN(min_cluster_size=15)
hdb.fit(embeddings)
hdb.labels_


# In[7]:


newlabel=pd.DataFrame(hdb.labels_)
newlabel.nunique()


# In[8]:


newlabel[newlabel==-1]=74
#print(newlabel)


# In[9]:


newnewnode['newskill'] = newlabel
#print(newnewnode)


# In[10]:


present=pd.read_csv('assist2017data3.csv')
#present['skillcluster']=newnewnode
present = present.merge(
    newnewnode[['problemId', 'newskill']],  
    on='problemId',                
    how='left'               # retain all rows of df2
)


# In[11]:


mean_val = np.round(present['newskill'].mean())
present['newskill'].fillna(mean_val,inplace=True)


# In[12]:


present['newskill']=present['newskill'].astype(int)
end_time=time.time()
all_time=end_time-start_time
print('all GNN time is',all_time)

