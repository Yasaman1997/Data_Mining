
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


theta = 0.8
samples = (np.random.random(1000) < theta) * 1


# In[3]:


samples


# In[4]:


theta_candidates = np.arange(0,1,0.001)


# In[5]:


N = np.sum(samples == 1)


# In[6]:


for theta_hat in theta_candidates:
    log_likelihood = np.log(theta_hat)*N + np.log(1-theta_hat)*(1000-N)
    print(theta_hat, log_likelihood)


# In[7]:


theta_candidates[np.argmax(np.log(theta_candidates)*N + np.log(1-theta_candidates)*(1000-N))]

