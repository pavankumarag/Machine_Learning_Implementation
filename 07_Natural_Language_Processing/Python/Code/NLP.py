#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:30:14 2017

@author: pavan.govindraj
"""
#Bag of words representation, NLP

import pandas as pd
import numpy as np
import matplotlib as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word)for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    #dataset['Review'][i] = review
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

#Splitting the dataset in to training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.20, 
                                                    random_state = 0)

#Fitting classifier to the dataset
#create a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predict the test set results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy (55+91)/200 = 73%