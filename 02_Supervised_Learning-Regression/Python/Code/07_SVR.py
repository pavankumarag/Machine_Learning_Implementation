#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:04:46 2016

@author: pavan.govindraj
"""

#SVR

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
#May be required for some dataset 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


#Fitting the  SVR regression model to the dataset
#create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

#Predicting the new value
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising the SVR regression results
plt.scatter(X, Y, color = 'red' )
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Truth or Bluff( Regression Model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising the SVR regression results(for higher resoulution and smooth curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red' )
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff( Regression Model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
