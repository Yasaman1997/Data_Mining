import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# read data
train = pd.read_csv('..//input//train.csv')
print(train.columns)



# Filling missing values
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Age'] = train['Age'].fillna(train['Age'].median())
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"] = train["Embarked"].fillna("S")


# Categorical to numerical
# Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

# convert from float to int
train['Fare'] = train['Fare'].astype(int)
train['Fare'] = train['Fare'].astype(int)
train['Age'] = train['Age'].astype(int)
train['Age'] = train['Age'].astype(int)
train['Embarked'] = train['Embarked'].astype(int)
train['Embarked'] = train['Embarked'].astype(int)
train['Sex'] = train['Sex'].astype(int)
train['Sex'] = train['Sex'].astype(int)

# pull data
train_y = train.Sex
predictor_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp']

# create trainnig prediction data
train_x = train[predictor_cols]

# model
my_model = RandomForestRegressor()
my_model.fit(train_x, train_y)



# Read the test data
test = pd.read_csv('test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_ages = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.
print(predicted_ages)

my_submission = pd.DataFrame({'Id': test.Id, 'Sex': predicted_ages})

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
