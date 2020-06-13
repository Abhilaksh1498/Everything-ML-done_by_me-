# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:03:01 2020

@author: MSI_PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\SVM_Gridsearch R, Py\train.csv')
y_train = train_dataset.iloc[:,1].values
train_dataset= train_dataset.drop(['Cabin','Survived','Ticket','Name'], axis = 1)
train_dataset = train_dataset.iloc[:,1:]

from sklearn.preprocessing import Imputer
imp = Imputer(strategy= 'mean', axis = 0) # axis =0 => cols, We have 263 nan in age
train_dataset.Age = imp.fit_transform(train_dataset.Age.values.reshape(-1,1)) # dataset.Age.isnull.sum() = 0
# We'll replace with mode for categorical variable embarked which has 2
train_dataset.Embarked=train_dataset.Embarked.fillna('S')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_dataset.Sex= le.fit_transform(train_dataset.Sex)
train_dataset.Embarked = le.fit_transform(train_dataset.Embarked)

train_dataset = pd.concat([train_dataset,pd.get_dummies(drop_first = True, data = train_dataset.Pclass, prefix= 'Pclass'),
                     pd.get_dummies(drop_first = True, data = train_dataset.Embarked, prefix= 'Embark')],
                     axis = 1).drop(['Pclass','Embarked'], axis =1)
X_train = train_dataset.iloc[:,:].values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# Adaboost
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV
param_grid = [{'learning_rate':[.5,1,2,3,4,5], 'n_estimators':[10,50,100,500,1000]}]
clf_ada = GridSearchCV(estimator=AdaBoostClassifier(),
                       scoring = 'accuracy', param_grid= param_grid, cv=5)
clf_ada.fit(X_train, y_train)
best_param_accuracy = clf_ada.best_params_
best_accuracy_ada = clf_ada.best_score_

from sklearn.model_selection import cross_val_score
f1_ada = cross_val_score(estimator=clf_ada.best_estimator_, X = X_train, y= y_train, cv=5, scoring= 'f1').mean()

# adaboost did prettry well Accuracy: .815, f1_score : .755

# Gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier
param_grid_gb = [{'n_estimators':[10,50,100,500,1000], 'learning_rate':[.1,.2,.3,.5,1,.09,.01],
                     'min_samples_split': [2,5,10,20,30]}]
clf_gb = GridSearchCV(estimator=GradientBoostingClassifier(),
                       scoring = 'accuracy', param_grid= param_grid_gb, cv=5)
clf_gb.fit(X_train, y_train)
best_accuracy_gb = clf_gb.best_score_

f1_gb = cross_val_score(estimator=clf_gb.best_estimator_, 
                         X = X_train, y= y_train, cv=5, scoring= 'f1').mean()
# Gradient Boosting even better: Accuracy = .84, F1 = .778

# let's predict titanic and submit on kaggle
test_dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\SVM_Gridsearch R, Py\test.csv')
test_dataset= test_dataset.drop(['Cabin','Ticket','Name'], axis = 1)
pass_id = test_dataset.PassengerId
test_dataset = test_dataset.iloc[:,1:]
test_dataset.Age = imp.transform(test_dataset.Age.values.reshape(-1,1)) # dataset.Age.isnull.sum() = 0
test_dataset.Sex= le.fit_transform(test_dataset.Sex)
test_dataset.Embarked = le.fit_transform(test_dataset.Embarked)
test_dataset = pd.concat([test_dataset,pd.get_dummies(drop_first = True, data = test_dataset.Pclass, prefix= 'Pclass'),
                     pd.get_dummies(drop_first = True, data = test_dataset.Embarked, prefix= 'Embark')],
                     axis = 1).drop(['Pclass','Embarked'], axis =1)
test_dataset.Fare = imp.fit_transform(test_dataset.Fare.values.reshape(-1,1))
X_test = sc_X.transform(test_dataset)

# Data ready for predictions
# Use gradientboost classifier
y_pred = pd.DataFrame(clf_gb.predict(X_test))
pred_df = pd.concat([pass_id, y_pred], axis =1)
predictions = open('predictions.csv','w')
pred_df.to_csv(path_or_buf = predictions, index =False)
