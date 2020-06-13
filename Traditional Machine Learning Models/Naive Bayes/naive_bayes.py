# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:52:50 2020

@author: MSI_PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# You include "r" before importing dataset from any random directory
dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Coding Practice!!\Machine Learning\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 18 - Naive Bayes\Social_Network_Ads.csv')
dataset = dataset.iloc[:,1:]

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape((-1,1))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0]= le.fit_transform(X[:,0])

# We dont need feature scaling here, since naive bayes doesn't have any distances or such involved


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# We'll first directly use Gausian kernel
# Then we'll split predictors as continuous and categorical features and multiply the probabilities to get the majority

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train.reshape((-1,)))
prob_predict = nb.predict_proba(X_test)
     
y_pred = nb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# First only the categorical using BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train[:,0].reshape((-1,1)), y_train.reshape((-1,)))
bnb_prob = bnb.predict_proba(X_test[:,0].reshape((-1,1)))

# Taking care of Continuous predictor cols
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train[:,1:], y_train.reshape((-1,)))
mnb_prob = mnb.predict_proba(X_test[:,1:])

# This is how we multiply corresponding elements of numpy array
net_prob = np.multiply(bnb_prob,mnb_prob)
y_pred_improv = []
for i in range(len(net_prob)):
       y_pred_improv.append(0 if net_prob[i][0]> net_prob[i][1] else 1)
np.asarray(y_pred_improv)
#np.asarray(y_pred_improv).reshape((-1,1))
# Precision Among the ones you classified as minority class how much were actually minority
# Recall Among the samples which were actually minority how many did u successfully label as minority
cm2 = confusion_matrix(y_test, y_pred_improv)
P_cm2 = cm2[1][1]/(cm2[0][1]+cm2[1][1])
R_cm2 = cm2[1][1]/(cm2[1][0]+cm2[1][1])
F_cm2 = 2*P_cm2*R_cm2/(P_cm2 + R_cm2) #.286

P_cm = cm[1][1]/(cm[0][1]+cm[1][1])
R_cm = cm[1][1]/(cm[1][0]+cm[1][1])
F_cm = 2*P_cm*R_cm/(P_cm + R_cm) #.86

#The improvement is not at all an improvement bad both in terms of accuracy and precision as well as recall
# The cm[i][j] is no of samples you classified in jth class but were actually in ith class
# Also note that minority class is taken as positive class (in def of precision and recall)

# We'll now just visualize the features to get an idea if they can really be fitted 
# to a gaussian distribution
# For this we'll use SEABORN library
import seaborn as sns

sns.distplot(dataset.iloc[:,2], hist = False,  
            kde = True, rug= False)
plt.xlabel("Estimated Salary")
plt.title('KDE plot only')
plt.show()

# This is used to map a function to all elements of numpy array
import math
def ln_func(n):
       return math.log10(n)
applies_log = np.vectorize(ln_func)

# Lets check if log version of features fit better to this Gaussian approximation
sns.distplot(applies_log(dataset.iloc[:,2].values), hist = False,  
            kde = True, rug= False)
plt.xlabel("Estimated Salary")
plt.title('KDE plot only')
plt.show()
# Somewhat better approximation but still not upto mark

