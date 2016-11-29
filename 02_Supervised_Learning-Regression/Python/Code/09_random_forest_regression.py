#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 21:57:11 2016

@author: pavan.govindraj
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:57:15 2016

@author: pavan.govindraj
"""


# Data preprocessing => import the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


#Splitting the dataset in to training set and test set
#No training set and test set splitting as max info is required 
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, 
                                                    random_state = 0)
'''

#Feature scaling
"""
#May be required for some dataset 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fitting the Random forest regression model to the dataset
#create your regressor here
from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,Y)

#Predicting the new value
y_pred = regressor.predict(6.5)

#Visualising the Random forest regression results
#As Random forest is non continuos, we need high resolution
'''
plt.scatter(X, Y, color = 'red' )
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Truth or Bluff( Regression Model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()'''

#Visualising the Random forest regression results(for higher resoulution and smooth curve)
#X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red' )
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff( Random Forest Regression Model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
