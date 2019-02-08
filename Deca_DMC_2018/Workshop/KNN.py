
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home="./data")


# In[2]:


print(mnist.data.shape)
print(mnist.target.shape)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


fig, array = plt.subplots(2, 5)

for i, index in enumerate(np.random.choice(mnist.data.shape[0], 10)):
    image = np.reshape(mnist.data[index], (28, 28))
    array[int(i/5), i%5].imshow(image, cmap='gray')
    array[int(i/5), i%5].set_title(mnist.target[index])
    
plt.show()


# # Split data

# In[5]:


import numpy as np

np.random.seed(123)


# In[6]:


#Shuffle data
permutation = np.random.permutation(mnist.data.shape[0])

X = mnist.data[permutation]
y = mnist.target[permutation]


# In[8]:


# Split data
N_train = 60000

X_train = X[:N_train]
y_train = y[:N_train]

X_test = X[N_train:]
y_test = y[N_train:]


# # KNN

# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[10]:


model = KNeighborsClassifier(n_neighbors=3)


# In[11]:


model=model.fit(X_train,y_train)


# In[12]:


y_hat = model.predict(X_test[:1000])


# # Evaluation

# In[13]:


from sklearn import metrics


# In[22]:


print("test accuracy:",metrics.accuracy_score(y_test[:1000],y_hat))


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

fig, array = plt.subplots(2, 5)
for i, index in enumerate(np.random.choice(X_test[:1000].shape[0], 10)):
    while y_hat[index] == y_test[index]:
        index = np.random.choice(X_test[:1000].shape[0])
    image = np.reshape(X_test[index], (28, 28))
    array[int(i/5), i%5].imshow(image, cmap='gray')
    array[int(i/5), i%5].set_title(y_hat[index])
plt.show()

