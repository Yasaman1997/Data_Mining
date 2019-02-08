
# coding: utf-8

# In[90]:


from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

get_ipython().magic('matplotlib inline')


# In[91]:


iris = datasets.load_iris()
print (iris.data)
print (iris.target)


# In[92]:


pd.set_option('display.max_rows',200)


# In[93]:


iris.keys()


# In[94]:


x=iris['data']


# In[95]:


df=pd.DataFrame(x)


# In[96]:


df.columns=iris['feature_names']
df['target']=iris['target']
df


# In[97]:


km=KMeans(n_clusters=3,max_iter=500000)


# In[98]:


km.fit(x)


# In[99]:


km.cluster_centers_


# In[100]:


km.labels_


# In[101]:


df['Kmeans predicted labels']=km.labels_


# In[102]:


df

