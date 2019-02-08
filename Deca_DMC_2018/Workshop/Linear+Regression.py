
# coding: utf-8

# In[38]:


import pandas as pd

table = pd.read_excel('./data/ToyotaCorolla.xls', sheet_name='data')


# In[39]:


table


# In[40]:


table.columns


# # Categorical to Numerical Conversion
# 

# In[41]:


print("Fules:",set(table['Fuel_Type']))


# In[42]:


print("Colors:",set(table['Color']))


# In[43]:



data = table[['Id', 'Model', 'Price', 'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM',
                  'HP', 'Met_Color', 'Automatic', 'CC', 'Doors',
                  'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight', 'Mfr_Guarantee',
                   'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
                   'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
                   'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
                   'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
                   'Radio_cassette', 'Parking_Assistant', 'Tow_Bar']].copy()


# In[44]:


data['Diesel'] = (table['Fuel_Type'] == 'Diesel') * 1
data['Petrol'] = (table['Fuel_Type'] == 'Petrol') * 1


# In[45]:


data


# In[46]:


for color_name in  sorted(list(set(table['Color'])))[1:]:
    data[color_name] = (table['Color'] == color_name) * 1


# In[47]:


new_table = data.copy()


# In[48]:


new_table.columns


# In[49]:


new_table.columns


# In[51]:


new_table.to_excel('./data/ToyotaCorolla_cat2num.xls', sheet_name='data')


# # Set Input and Output

# In[52]:


new_table


# In[60]:



x = new_table[['Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM',
       'HP', 'Met_Color', 'Automatic', 'CC', 'Doors', 'Cylinders', 'Gears',
       'Quarterly_Tax', 'Weight', 'Mfr_Guarantee', 'BOVAG_Guarantee',
       'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2', 'Airco',
       'Automatic_airco', 'Boardcomputer', 'CD_Player', 'Central_Lock',
       'Powered_Windows', 'Power_Steering', 'Radio', 'Mistlamps',
       'Sport_Model', 'Backseat_Divider', 'Metallic_Rim', 'Radio_cassette',
       'Parking_Assistant', 'Tow_Bar', 'Diesel', 'Petrol', 'Silver',
       'Violet', 'Grey', 'Yellow', 'Blue', 'Green', 'Red', 'Black',
       'White']]


# In[61]:


y=new_table['Price']


# In[62]:


x = x.as_matrix()
y = y.as_matrix()


# In[64]:


x.shape , y.shape


# In[65]:


len(x)


# # Random Generator

# In[68]:


import numpy as np   


# In[69]:


np.random.seed(100)


# In[71]:


#Shuffle data
permutation = np.random.permutation(len(X))
X = X[permutation]
y = y[permutation]


# In[72]:


# Split data
N_train = int(len(X) * 4 / 5)

X_train = X[:N_train]
y_train = y[:N_train]

X_test = X[N_train:]
y_test = y[N_train:]


# # Linear Regression Using Scikit-Learn

# In[75]:


from sklearn import linear_model


# In[76]:


model = linear_model.LinearRegression()
model.fit(X_train,y_train)


# In[78]:


model.coef_   
#The targets are the values you want to predict. The ridge regression can in fact predict more values for each instance,
#not only one. The coef_ contain the coefficients for the prediction of each of the targets.
#It is also the same as if you trained a model to predict each of the targets separately.

