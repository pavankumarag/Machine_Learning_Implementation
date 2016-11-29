#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:30:26 2016

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

#Fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # we can use more degree for precise
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visualising the linear regression
plt.scatter(X, Y, color = 'red' )
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising the polynomial regression
plt.scatter(X, Y, color = 'red' )
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Improving the polynomial prediction, making the curve smoooth than lines
plt.scatter(X, Y, color = 'red' )
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Predicting the new value using linear regression
lin_reg.predict(6.5)

#Predicting the new value using polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

