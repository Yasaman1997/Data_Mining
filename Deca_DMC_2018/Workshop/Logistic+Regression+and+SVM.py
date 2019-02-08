
# coding: utf-8

# # Logistic Regression and SVM

# In[2]:


from sklearn.datasets import fetch_mldata

mnist =fetch_mldata('mnist original',data_home="./data")


# In[3]:


mnist.data.shape


# In[4]:


mnist.target.shape


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

fig, array = plt.subplots(2, 5)
for i, index in enumerate(np.random.choice(mnist.data.shape[0], 10)):
    image = np.reshape(mnist.data[index], (28, 28))
    array[int(i/5), i%5].imshow(image, cmap='gray')
    array[int(i/5), i%5].set_title(mnist.target[index])
plt.show()


# # Split data

# In[9]:


import numpy as np

np.random.seed(123)

#Shuffle data
permutation = np.random.permutation(mnist.data.shape[0])
X = mnist.data[permutation]
y = mnist.target[permutation]

# Split data
N_train = 60000

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]


# # SVM

# In[10]:


from sklearn.svm import LinearSVC

model = LinearSVC(multi_class='ovr')
model.fit(X_train, y_train)


# # Evaluation
# 

# In[18]:


y_hat = model.predict(X_test)


# In[23]:


from sklearn import metrics

print("test accuracy:", metrics.accuracy_score(y_test, y_hat))

