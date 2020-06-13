# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:26:11 2020

@author: MSI_PC
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\SVM_Gridsearch R, Py\train.csv')
test_dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\SVM_Gridsearch R, Py\test.csv')

y_train = train_dataset.iloc[:,1].values

# Cleaning data
       # We usually remove cols if they have >=70% of missing data (here 77% of cabin column missing)
       # Else for categorical ones we introduce a new category
       # We will replace age with mean (for numerical) age for both test and train
       # for categorical (better to use mode or new category)
train_dataset= train_dataset.drop(['Cabin'], axis = 1)
test_dataset= test_dataset.drop(['Cabin'], axis = 1)
train_dataset= train_dataset.drop(['Survived'], axis = 1)
dataset = pd.concat([train_dataset, test_dataset], keys = ["X_train", "X_test"])
dataset = dataset.drop(['PassengerId'], axis =1)
dataset = dataset.drop(['Ticket'], axis =1)
from sklearn.preprocessing import Imputer
imp = Imputer(strategy= 'mean', axis = 0) # axis =0 => cols, We have 263 nan in age
dataset.Age = imp.fit_transform(dataset.iloc[:,3].values.reshape(-1,1)) # dataset.Age.isnull.sum() = 0
# Looks like we have 3 nulls left, 1 in fare, 2 in embark We can remove these rows
dataset = dataset.dropna(axis = 0) # 0=> rows, 1=> cols
# Label encode all the categorical/binary features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.Sex= le.fit_transform(dataset.Sex)
dataset.Embarked = le.fit_transform(dataset.Embarked)

# For the last time I am searching for one hot encoding
# get_dummies function on pandas
dataset = pd.concat([dataset,pd.get_dummies(drop_first = True, data = dataset.Pclass, prefix= 'Pclass'),
                     pd.get_dummies(drop_first = True, data = dataset.Embarked, prefix= 'Embark')],
                     axis = 1).drop(['Pclass','Embarked'], axis =1)

dataset=dataset.drop(['Name'], axis =1)

# You first split and then do feature scaling
# fit_transform to X_train, only transform to X_test
X_train = dataset.iloc[:891,:].values
X_test = dataset.iloc[891:,:].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#np.random.shuffle(X_train)
#cv = X_train[:180,:]
#X_train = X_train[180:,:]
# Instead of standard test train we'll use kfold cross validation in conjunction with GridSearch
# We'll only use X_train and y_train for this
from sklearn.svm import SVC


# We'll tune the hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = [{'kernel':['linear'], 'C':[.1,1,10,100]}, {'kernel':['rbf'], 'C':[.1,1,10,100],
               'gamma':[.09,.1,.2,.3,.5,.4,.6,.7,.8,.9]}]
clf = GridSearchCV(estimator=SVC(), scoring = 'accuracy', param_grid= param_grid, cv=5)
clf.fit(X_train, y_train)
best_param_accuracy = clf.best_params_
best_accuracy = clf.best_score_
                  

#from sklearn.metrics import SCORERS
#scoring = SCORERS # these are variousscoring metrics that can be used
scoring = ['f1','accuracy']

#let's verify accuracy score using cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator=clf.best_estimator_, X = X_train, y= y_train, cv=5, scoring= 'accuracy')
# the score.mean() is exactly same as best_accuracy

# See which model works well for f_score 
clf_f1 = GridSearchCV(estimator=SVC(), scoring = 'f1', param_grid= param_grid, cv=5)
clf_f1.fit(X_train, y_train)
best_param_f1 = clf_f1.best_params_
best_f1 = clf_f1.best_score_
results_summary = clf_f1.cv_results_

# Verify using CV
score_rbf = cross_val_score(estimator=clf_f1.best_estimator_, X = X_train, y= y_train, cv=5, scoring= 'f1')

# Creating a dataframe
summary = pd.DataFrame.from_dict(results_summary)
summary = summary.iloc[:,[7,13,15]]
summary = pd.concat([summary, pd.DataFrame(clf.cv_results_['mean_test_score'])], axis = 1)
# From summary we can see that linear model is not classifying properly (skewed classification)
y_pred_linear = clf.best_estimator_.predict(X_test)
y_pred_rbf = clf_f1.best_estimator_.predict(X_test)

# Hence better model is rbf one

# As a fun lets plot variance as a function of k
cv_f1_scores = []
cv_acc_scores = []
k = []
for i in range(2,254,6):
       k.append(i)
       cv_f1_scores.append(cross_val_score(estimator=clf_f1.best_estimator_, 
                       X = X_train, y= y_train, cv=i, scoring= 'f1').var())
       cv_acc_scores.append(cross_val_score(estimator=clf_f1.best_estimator_, 
                       X = X_train, y= y_train, cv=i, scoring= 'accuracy').var())
k.pop(-1)
plt.plot(k,cv_f1_scores, 'bo-', label ='f1_scores' )
plt.plot(k,cv_acc_scores, 'g+-', label ='accuracy_scores' )
plt.legend()
plt.xlabel('CV (k)')
plt.ylabel('Variance')
plt.title('Variation of acc and f1 scores with K')
plt.show()

# From here we can clearly see that variance increases as k increases
       