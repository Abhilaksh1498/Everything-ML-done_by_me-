# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:08:49 2020

@author: MSI_PC
"""
import pandas as pd
import numpy as np
f = open(r"C:\Users\MSI_PC\Desktop\Chatbot\Text Classification\sentiment labelled sentences\amazon_cells_labelled.txt",'r')
lines = f.readlines()   # returns a list of all lines separated by \n
lines = list(dict.fromkeys(lines))  # remove any duplicates
labels = []
reviews = []
for l in lines:
       cleaned_line = l.rstrip('\n').rstrip() #removed \n, any space between label and \n
       labels.append(cleaned_line[-1])
       reviews.append(cleaned_line[:-1].rstrip())
       
labels = np.array(labels) 
reviews = np.array(reviews)     
dataset = pd.concat([pd.DataFrame(reviews), pd.DataFrame(labels)], axis = 1)
f.close() 
dataset.columns = ['Reviews', 'Labels']
dataset.to_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\Text Classification\sentiment labelled sentences\amazon_cells_labelled.tsv', sep = '\t',
               index = False)

try_dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\Chatbot\Text Classification\sentiment labelled sentences\amazon_cells_labelled.tsv', 
                          delimiter= '\t')

