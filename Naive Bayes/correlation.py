# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:18:06 2020

@author: MSI_PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# You include "r" before importing dataset from any random directory
dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Coding Practice!!\Machine Learning\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 18 - Naive Bayes\Social_Network_Ads.csv')
dataset = dataset.iloc[:,1:]

# We can also see how well the independent approximation holds using the correlation matrix
# Use corr() function in pandas dataframe object
corr_matrix = dataset.iloc[:,:-1].corr()
# Corr b/w age and estimated salary is .15 so independent approx is fairly good

# Properties of Correlation and Covariance
#1.Correlation = [-1,1], Covariance = R
#2. Correlation - unitless but covar is not
#3. Corr - tells the strength as well as direction |cov|>.8 => strong dependency
#4. Cov only tells direction its magnitude is not an indicator of anything
#5. Independent variables have 0 corr and cov
#6. Corr = cov/sigma(x)*sigma(y)
#7. The Cij = Covariance of column i and column j