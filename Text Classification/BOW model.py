# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:40:58 2020

@author: MSI_PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\Text Classification\corpus.csv',
                      encoding='latin-1')  
# You can use latin-1 encoding whenever there is error loading file
# its always better to use a tsv file instead of csv
# because reviews may contain commas which can mess up dataset (though it worked here)
# Using this dataset[dataset.duplicated()] we can check for duplicates

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.label = le.fit_transform(dataset.label)
# 1 = positive and vice versa
corpus = []
# Cleaning the reviews
#1. remove punctuations and other characters
import re
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
for i in range(0,len(dataset.text)):
       review = dataset.text[i]
       review = re.sub(pattern= '[^a-zA-Z]', repl=' ', string= review)
       #2. lowercase
       review = review.lower()
       #3. remove words that are not necessary eg. the/this/... using stopwords list
       #4. stem the remaining words to get only the root words
       review = review.split()
       review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
       review = ' '.join(review)
       corpus.append(review)

# The data is now ready to create a BOW model (sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features = 750)      
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# We'll compare naive bayes, svm and boosting classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# SVM
from sklearn.svm import SVC
param_svm = [ {'kernel':['rbf'], 'C':[.1,1,10,100]}]
clf_svm = GridSearchCV(estimator=SVC(), scoring = 'accuracy', param_grid= param_svm, cv=5)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.best_estimator_.predict(X_test)  # Acc = 82.4, f1 = .83
# Without tuning acc was around 59 percent

# Gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier
param_grid_gb = [{'n_estimators':[100,500,1000], 'learning_rate':[.1,1,.01]}]
clf_gb = GridSearchCV(estimator=GradientBoostingClassifier(),
                       scoring = 'accuracy', param_grid= param_grid_gb, cv=3, n_jobs = -1)
clf_gb.fit(X_train, y_train)
#acc_gb = clf_gb.best_score_
classifier = GradientBoostingClassifier(learning_rate = 1, n_estimators = 500)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)  # Acc = 81.95, f1 = .819

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Using another vectoriser Tf-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idfv = TfidfVectorizer(max_features = 250, lowercase = False)
X2= tf_idfv.fit_transform(corpus)
X2 = X2.toarray()

from sklearn.model_selection import train_test_split
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier
param_grid_xg = [{
     "max_depth"        : [ 8, 7],
    # "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.001,.01]
     }]
grid = GridSearchCV(XGBClassifier(),
                    param_grid_xg, n_jobs=-1,
                    scoring="accuracy",
                    cv=3)
grid.fit(X2_train,y_train)
clf_xgb = grid.best_estimator_
y_pred_xgb = clf_xgb.predict(X2_test)  #80.45 but with 250 features

# Hashing Vectorizer
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=2**9, lowercase = False)
X3 = hv.fit_transform(corpus)
X3 = X3.toarray()
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size = 0.2, random_state = 0)

param_grid_gb = [{'n_estimators':[100,500,1000], 'learning_rate':[.1,.1,.01],
                     'min_samples_split': [2,10,50]}]
clf_gb = GridSearchCV(estimator=GradientBoostingClassifier(),
                       scoring = 'accuracy', param_grid= param_grid_gb, cv=3)
clf_gb.fit(X3_train, y_train)
best_accuracy_gb = clf_gb.best_score_
