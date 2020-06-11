# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:50:34 2019

@author: MSI_PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
N = 10000
d = 10
ad_selected = []
num_rewards0 = [0]*d
num_rewards1= [0]*d
for i in range(0,N):
       theta = []
       for j in range(0,d):
              theta.append(random.betavariate(num_rewards1[j]+1, num_rewards0[j]+1))
       next_index = theta.index(max(theta))
       ad_selected.append(next_index+1)
       if dataset.values[i,next_index]==1:
              num_rewards1[next_index]+=1
       else:
              num_rewards0[next_index]+=1
total_reward = sum(num_rewards1)
plt.hist(ad_selected, color='red')