import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
N = 10000
d = 10
ad_selected = []
rewards = [0]*d
no_of_selections = [0]*d
total_reward =0
for i in range(0,N):
       if i<=9:
              rewards[i] += dataset.values[i,i]
              no_of_selections[i]+=1
              ad_selected.append(i+1)
       else:
              upper_bound =[]
              for j in range(0,d):
                     upper_bound.append(rewards[j]/no_of_selections[j]+ math.sqrt(1.5*math.log(i+1)/no_of_selections[j]))
              next_index = upper_bound.index(max(upper_bound))
              no_of_selections[next_index]+=1
              rewards[next_index]+= dataset.values[i,next_index]
              ad_selected.append(next_index+1)
total_reward = sum(rewards)
plt.hist(ad_selected)
plt.xlabel('Ad number')
plt.ylabel('No of selections')
plt.title('Histogram')
plt.show()

actual_max = sum(dataset.iloc[:,4].values)      
