#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:32:06 2016

@author: pavan.govindraj
"""

#simple linear regression
# Data preprocessing => import the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values  #independent variable/matric
Y = dataset.iloc[:, 1].values    #dependent variable/vector


#Splitting the dataset in to training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 1/3, 
                                                    random_state = 0)

#Feature scaling
"""
#May be required for some dataset 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fitting simple linear regression to the training set 
#linear equation Y= a *X + b. where Y -> Dependent Variable; a -> slope
#b -> intercept ; X -> Independent Variable
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#regressor.score(X_train, y_train)

#Equation coefficient and Intercept
print('Coefficient: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue' )
plt.title("Salary VS. Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue' )
plt.title("Salary VS. Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


