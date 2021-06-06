#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle as pkl


# In[3]:


A=pd.read_csv("50_Startups.csv")


# In[4]:


A


# In[5]:


Y=A[['PROFIT']]
X=A[['RND',"MKT"]]

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(X,Y)


# In[6]:


pkl.dump(model,open("model.pkl","wb"))
model=pkl.load(open("model.pkl","rb"))


# In[ ]:




