#This model predicts the salary of the employ based on experience using simple linear regression model.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('SalaryData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


regresso = LinearRegression()
regresso.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regresso.predict(X_test)

# Saving model to disk
pickle.dump(regresso, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[4.9]]))
