
# coding: utf-8

# In[1]:


# First let's import the dataset, using Pandas.
import pandas as pd

train = pd.read_csv("train.csv")    # make sure you're in the right directory if using iPython!
test = pd.read_csv("test.csv") 

train.head()             # ignore the first column, it's how I split the data.


# In[2]:


from sklearn.ensemble import RandomForestClassifier



# however, are data has to be in a numpy array in order for the random forest algorithm to except it!
cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
colsRes = ['class']
trainArr = train.as_matrix(cols)    # training array
trainRes = train.as_matrix(colsRes) # training results



## Training!

rf = RandomForestClassifier(n_estimators=100)    # 100 decision trees is a good enough number
rf.fit(trainArr, trainRes)          # finally, we fit the data to the algorithm!!! :)

# note - you might get an warning saying you entered a 2 column vector..ignore it.


# In[3]:


## Testing!

# put the test results in the same format!
testArr = test.as_matrix(cols)

results = rf.predict(testArr)

# something I like to do is to add it back to the dataframe, so I can compare side-by-side
test['predictions'] = results
test.head()

