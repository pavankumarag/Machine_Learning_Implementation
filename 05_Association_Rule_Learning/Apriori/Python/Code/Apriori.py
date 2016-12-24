#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 00:05:17 2016

@author: pavan.govindraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
#Training Apriori on the dataset 
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, 
                min_length = 2) #support is cal as 3*7/7501

#Visualising the results
results = list(rules)
for i in range(0,11):
    lhs = str(results[i][2]).split('=')[1]
    rhs = str(results[i][2]).split('=')[2]
    lhs = lhs[0:lhs.rfind(',')]
    lhs = lhs.replace('frozenset','')
    rhs = rhs[0:rhs.rfind(',')]
    rhs = rhs.replace('frozenset','')
    print("%r -> %r" %(lhs,rhs))