# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:09:46 2019

@author: MSI_PC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
       kmeans = KMeans(n_clusters=i, random_state=0)
       wcss.append(kmeans.fit(X).inertia_)
plt.plot(range(1,11), wcss)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans= kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], color = 'red')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], color = 'blue')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], color = 'green')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], color = 'magenta')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], color = 'cyan')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='yellow')
plt.show()
