
# coding: utf-8

# In[68]:


from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

get_ipython().magic('matplotlib inline')


# In[69]:


from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from numpy.random import RandomState
from itertools import cycle
import pylab as pl


# In[70]:


iris = load_iris()
X = iris.data
y = iris.target


# In[71]:


pca = PCA(n_components=2, whiten=True).fit(X)

X_pca = pca.transform(X)


# In[72]:


rng = RandomState(42)

kmeans = KMeans(n_clusters=3, random_state=rng).fit(X_pca)


# In[73]:


np.round(kmeans.cluster_centers_, decimals=2)


# In[74]:


kmeans.labels_[:20]


# In[75]:


kmeans.labels_[-20:]


# In[76]:


class clustering:
    def __init__(self):
            self.plot(load_iris().data)

    def plot(self, X):
            pca = PCA(n_components=2, whiten=True).fit(X)
            X_pca = pca.transform(X)
            kmeans = KMeans(n_clusters=3, random_state=RandomState(42)).fit(X_pca)
            #plotting
            plot_2D(X_pca, kmeans.labels_, ["c0", "c1", "c2"])

def plot_2D(data, target, target_names):
        colors = cycle('rgbcmykw')
        target_ids = range(len(target_names))
        pl.figure()
        for i, c, label in zip(target_ids, colors, target_names):
            pl.scatter(data[target == i, 0], data[target == i, 1],
            c=c, label=label)
            
        pl.legend()
        pl.show()


if __name__ == '__main__':
	c = clustering()

