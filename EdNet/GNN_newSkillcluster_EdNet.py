#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Begin
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as plt
from matplotlib import pyplot
import time
from itertools import permutations
from scipy.spatial.distance import cosine


# In[2]:


nodes=pd.read_csv('./questions.csv')
print(nodes)


# In[3]:


node1=nodes[['question_id','bundle_id','explanation_id','part','tags']]
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
label_encoder = LabelEncoder()

node1['bundle_id'] = label_encoder.fit_transform(node1['bundle_id'])

node1['explanation_id'] = label_encoder.fit_transform(node1['explanation_id'])
node1['tags'] = label_encoder.fit_transform(node1['tags'])
newnode=node1.drop(columns=['question_id','bundle_id'])
y = node1[['question_id','bundle_id']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(newnode)
print(X_scaled)


# In[4]:


X=pd.DataFrame(X_scaled)
X.columns=['explanation_id','part','tags']
newnewnode=pd.concat([X,y],axis=1)
print(newnewnode)


# In[5]:


newnewnode.to_csv('ednetnodes.csv',index=None)


# In[14]:


#cosine-similarity
# skill grouping
result = []
feature_columns = [col for col in newnewnode.columns 
                   if col not in ['question_id', 'bundle_id']]
for skill, group in newnewnode.groupby('bundle_id'):
    # all sole ids 
    ids = group['question_id'].unique().tolist()
    print(f"Processing bundle_id {skill}, ids: {ids}")
    #print(ids)
    
    # combination
    pairs = list(permutations(ids, 2))
    #print(pairs)
    # cosine similarity
    for src, d in pairs:
        #scr_types = newnewnode.loc[newnewnode["question_id"] == scr].values.flatten()
        src_features = newnewnode.loc[newnewnode['question_id'] == src, feature_columns].values.flatten()
        #scr_types = newnewnode.loc[newnewnode['question_id'] == scr, newnewnode.columns != 'question_id'].values.flatten()
        #scr_types1 = scr_types
        #print(scr_types)
        #d_types = newnewnode.loc[newnewnode["question_id"] == d].values.flatten()
        dst_features = newnewnode.loc[newnewnode['question_id'] == d, feature_columns].values.flatten()
        #d_types = newnewnode.loc[newnewnode['question_id'] == d, newnewnode.columns != 'question_id'].values.flatten()
        #d_types1 = d_types.reshape(1,-1)
        #print(d_types)
        src_features = src_features.astype(float)
        dst_features = dst_features.astype(float)
        cos_distance = cosine(src_features, dst_features)
        similarity = 1 - cos_distance
        
        result.append({
            'Src': src,
            'Dst': d,
            'weight': similarity
            #'skill': skill  # retain skill
        })

# DataFrame
new_df = pd.DataFrame(result)

# validate
print(new_df)


# In[15]:


ids=newnewnode['question_id'].values
ids_map = {i: k for i, k in enumerate(ids)}
print(ids_map)


# In[16]:


newnewnode['question_id'] = newnewnode['question_id'].map(ids_map).fillna(newnewnode['question_id'])


# In[17]:


new_df['Src'] = new_df['Src'].map(ids_map).fillna(new_df['Src'])
new_df['Dst'] = new_df['Dst'].map(ids_map).fillna(new_df['Dst'])
print(new_df)


# In[18]:


new_df.to_csv('edgeednetinfo.csv',index=None)


# In[2]:


import pandas as pd
new_df=pd.read_csv('edgeednetinfo.csv')
newnewnode=pd.read_csv('ednetnodes.csv')


# In[3]:


# building graphs
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
start=time.time()
dgl_src = torch.from_numpy(new_df['Src'].to_numpy())
dgl_dst = torch.from_numpy(new_df['Dst'].to_numpy())
#print(dgl_src,dgl_dst, len(ids))
ids=newnewnode['question_id'].values
g = dgl.graph((dgl_src, dgl_dst), num_nodes=len(ids))
node_feature = torch.from_numpy(newnewnode.drop(columns=['question_id']).to_numpy())
g.ndata["feat"] = node_feature
g.edata['weight']= torch.tensor(new_df['weight'].values)

# validate
print("图结构:", g)
print("特征矩阵:\n", g.ndata["feat"], g.edata['weight'])


# In[4]:


# Final GNN construction and training
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
    # positive pairs
    src_emb = embeddings[edge_src]
    dst_emb = embeddings[edge_dst]
    pred_similarity = F.cosine_similarity(src_emb, dst_emb, dim=1)  # [num_edges]
    
    # MSE
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

            
# initial
model = GNN(in_feats=4, hidden_size=60, out_feats=190) 
g = dgl.add_self_loop(g)

# training
train(g, model, node_feature)


# In[5]:


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


# In[6]:


#derive final embedding
model.eval()
with torch.no_grad():
    embeddings = model(g, node_feature)
    print("Learned embeddings shape:", embeddings.shape)


# In[7]:


from sklearn.cluster import HDBSCAN
hdb = HDBSCAN(min_cluster_size=30)
hdb.fit(embeddings)
hdb.labels_


# In[8]:


newlabel=pd.DataFrame(hdb.labels_)
newlabel.nunique()


# In[9]:


newlabel[newlabel==-1]=80
#print(newlabel)


# In[10]:


newnewnode['newskill'] = newlabel
#print(newnewnode)


# In[11]:


present=pd.read_csv('mergenewall.csv')
#present['skillcluster']=newnewnode
present = present.merge(
    newnewnode[['question_id', 'newskill']],  
    on='question_id',                # merging keys
    how='left'               # retain rows for df2
)
#print(present)


# In[12]:


present = present.drop(columns=['Unnamed: 0'])
#print(present)


# In[13]:


present.isnull().sum()
end=time.time()
all_time=end-start
print('all time is',all_time)

