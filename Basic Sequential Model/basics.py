# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:01:53 2020

@author: MSI_PC
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv')
dataset = dataset.iloc[:,3:]

y = dataset.iloc[:,-1].values
X = dataset.iloc[:,:-1]

# No need for imputer on entire dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.Geography=le.fit_transform(X.Geography)
X.Gender=le.fit_transform(X.Gender)
X = pd.concat([X,pd.get_dummies(drop_first = True, data = X.Geography, prefix= 'Geography')],
                     axis = 1).drop(['Geography'], axis =1)

# Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# We always do feature scaling in every DL model
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# We'll build a NN using keras library
import keras
from keras import Sequential
from keras.layers import Dense, Dropout

# let's build a baseline dense model with 2 hidden layer and nodes = mean(ip,op)
clf = Sequential() 
clf.add(Dense(20, input_dim=11, activation='relu'))
    # after the first layer, you don't need to specify
    # the size of the input anymore
clf.add(Dense(16, activation='relu'))
clf.add(Dense(1, activation='sigmoid'))
# softmax is used for multiclass classification
# an instance of optimizer of class can be passed instead of string [it supports hyper pramaeter tuning]
# passing string version doesn't support changing parameters
clf.compile(optimizer = keras.optimizers.Nadam(), metrics =['accuracy'], loss='binary_crossentropy')

# Training the model
history = clf.fit(x = X_train, y= y_train, batch_size = 850, validation_split= .2, epochs= 1600)

score_history = history.history
y_pred_vanilla = clf.predict_classes(X_test)

plt.plot(range(1,1601), score_history['loss'], label = 'Train Loss')
plt.plot(range(1,1601), score_history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.show()
plt.plot(range(1,1601), score_history['accuracy'], label = 'Train Accuracy')
plt.plot(range(1,1601), score_history['val_accuracy'], label = 'Validation Accuracy')
plt.legend()
plt.xlabel('No of Epochs')
plt.ylabel('Accuracies')
plt.show()

# 175 roughly seems to be optimal no of epochs
# Lets increase batch size and we should see loss decrease more than sgd for some no of epochs
history = clf.fit(x = X_train, y= y_train, batch_size = 8500, validation_split= 0.0, epochs= 250)
score_history_bgd = history.history   #batch grad descent
plt.plot(range(1,1601), score_history_sgd['loss'], label = 'SGD train Loss')
plt.plot(range(1,1601), score_history_bgd['loss'], label = 'BGD train Loss')
plt.legend()
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.show()
#Let's see the optimal no of epochs
plt.plot(range(1,1601), score_history_bgd['loss'], label = 'Train Loss')
plt.plot(range(1,1601), score_history_bgd['val_loss'], label = 'Validation Loss')
plt.legend()
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.title('Batch Gradient Descent')
plt.show()
# Its around 250  #Accuracy 85.4%, f1 = .57

#Comparing itwith the best non deep learning model - Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
param_grid_gb = [{'n_estimators':[50,100,500,1000], 'learning_rate':[.1,.5,1,.01],
                     'min_samples_split': [2,10,20,30]}]
clf_gb = GridSearchCV(estimator=GradientBoostingClassifier(),
                       scoring = 'accuracy', param_grid= param_grid_gb, cv=5, n_jobs = -1)
clf_gb.fit(X_train, y_train)
#It outperformed in both accuracy : 87% and f1 : .626
