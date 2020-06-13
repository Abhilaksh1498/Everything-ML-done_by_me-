import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder


raw_dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\RH intern task\question2\ecommerce_sample_dataset.csv')
dataset=raw_dataset.drop(['image','crawl_timestamp', 'product_url'], axis = 1)
dataset=dataset.drop(['uniq_id','product_name','pid',], axis = 1)

# To check if product rating == overall rating
check_arr = (dataset.product_rating == dataset.overall_rating).to_numpy()
# all elements are True => we can remove either of the 2 columns
dataset=dataset.drop(['overall_rating'], axis =1)

# Note that description of item has quite un-necessary info and the useful info
# is already captured in cols product_specification, category tree, brand
# Hence we can remove description column as well
dataset=dataset.drop(['description'], axis =1) 

# We can also remove any duplicated rows
duplicates = dataset[dataset.duplicated()]
dataset = dataset.drop(labels = duplicates.index, axis = 0)
raw_dataset = raw_dataset.drop(labels = duplicates.index, axis = 0)
summary_dataset = dataset.isnull().sum()
# Note that we can replace nan entries of retail & discounted price with average of that column
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy= 'mean')
dataset.retail_price = imp.fit_transform(dataset.retail_price.to_numpy().reshape(-1,1))
dataset.discounted_price = imp.fit_transform(dataset.discounted_price.to_numpy().reshape(-1,1))

####
raw_dataset.retail_price = imp.fit_transform(raw_dataset.retail_price.to_numpy().reshape(-1,1))
raw_dataset.discounted_price = imp.fit_transform(raw_dataset.discounted_price.to_numpy().reshape(-1,1))

category_tree = []
for x in dataset.product_category_tree:
       x = x[2:-2]
       category_tree.append(x.split(" >>"))
category_tree = np.array(category_tree)
broadest_categories = []
category_less_index = []
i =0
for x in category_tree:
       if len(x)>1:
              broadest_categories.append(x[0])
       else:
              category_less_index.append(i)
       i+=1
              
# len(broadest_categories) = 18877
# len(dataset) [after removing duplicates] = 19204
# Categories for 327 training samples are missing, and since 327<<20000 we can delete those rows
plt.hist(broadest_categories)
plt.show()

##########################################################################
raw_dataset.reset_index(drop=True, inplace=True)

dataset.reset_index(drop=True, inplace=True)
dataset.drop(category_less_index,inplace = True)
raw_dataset.drop(category_less_index,inplace = True)
raw_dataset.reset_index(drop=True, inplace=True)

dataset.reset_index(drop=True, inplace=True)
categories = pd.DataFrame(broadest_categories)


##########################################################################
# replacing category tree with categories column
dataset = pd.concat([dataset, categories], axis = 1)
dataset.drop(['product_category_tree'],axis =1,inplace = True)
dataset = dataset.rename(columns = {0 : 'categories'})
# Brand column has around ~5500 empty entries so we'll drop that column
dataset.drop(['brand'],axis =1,inplace = True)
# As we already figured out categories, we might as well skip product_specification column
# to prepare baseline model
dataset.drop(['product_specifications'],axis =1,inplace = True)
dataset.drop(['product_rating'],axis =1,inplace = True)
# Tranform is_FK_advantage and categories column as categorical features to one hot encoded vectors
le = LabelEncoder()
dataset.is_FK_Advantage_product = le.fit_transform(dataset.is_FK_Advantage_product)
dataset.categories = le.fit_transform(dataset.categories)

# One hot Encoding of categories
dataset = pd.concat([dataset,pd.get_dummies(drop_first = True, data = dataset.categories, prefix= 'Productclass')],
                     axis = 1)
dataset.drop(['categories'], axis =1, inplace = True)

# Normalising the data so that retail/discounted price dont dominate other columns
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
dataset = sc_X.fit_transform(dataset)

#################################################################################
# We'll use cosine similarity instead of eucledian distance due to high dimensions and sparsity
import sklearn, operator
from sklearn.metrics.pairwise import cosine_similarity
def ComputeDistance(a, b):
       # a,b are vectors
       similarity = sklearn.metrics.pairwise.cosine_similarity(a, b, dense_output=False)
       return similarity

def getNeighbors(prodID, K= 5):
# prodID is any index from dataset/raw_dataset     
    distances = []
    movieID = prodID
    for productsID in range(0,len(dataset)):
        if (productsID != movieID):
            dist = ComputeDistance(dataset[movieID].reshape(1,-1), dataset[productsID].reshape(1,-1))
            distances.append((productsID, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return raw_dataset.iloc[neighbors,:]

###### this returns the most similar products from original dataset
    # your input to a function is a product index (among 18777 elements in dataset)

## generating some examples
eg_20 = getNeighbors(20)   #most similar products to index 20
eg_25 = getNeighbors(25)
eg_111 = getNeighbors(111)
eg_1201 = getNeighbors(1201)
eg_17000 = getNeighbors(17000)
submission = pd.concat([eg_20, eg_25,eg_111, eg_1201, eg_17000 ], axis = 0)
submission.to_csv(path_or_buf = r'C:\Users\MSI_PC\Desktop\RH intern task\question2\sample_submission_examples.csv', 
                  index =False)



############################################################################# BOW MODEL I USED 
# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Sun Apr 19 22:15:50 2020
# 
# @author: MSI_PC
# """
# 
# # To figure out the classes for 327 samples for which we dont have explicit category tree 
# #we can use BOW model on the remaining 18877 samples for which we know
# text = []
# labels = []
# X_test = []
# for x in category_tree:
#        if len(x)>1:
#               text.append(x[-1])
#               labels.append(x[0])
#        else:
#               X_test.append(x[0])
# bow_dataset = pd.concat([pd.DataFrame(np.array(text)), pd.DataFrame(np.array(labels))], axis =1)
# bow_dataset.columns = ['text', 'labels']
# import re
# import nltk
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# from nltk.corpus import stopwords
# corpus = []
# for i in range(0,len(bow_dataset.text)):
#        review = ' '.join(bow_dataset.text[i])
#        review = re.sub(pattern= '[^a-zA-Z]', repl=' ', string= review)
#        #2. lowercase
#        review = review.lower()
#        #3. remove words that are not necessary eg. the/this/... using stopwords list
#        #4. stem the remaining words to get only the root words
#        review = review.split()
#        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
#        review = ' '.join(review)
#        corpus.append(review)
# for x in X_test:
#        review = x
#        review = re.sub(pattern= '[^a-zA-Z]', repl=' ', string= review)
#        #2. lowercase
#        review = review.lower()
#        #3. remove words that are not necessary eg. the/this/... using stopwords list
#        #4. stem the remaining words to get only the root words
#        review = review.split()
#        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
#        review = ' '.join(review)
#        corpus.append(review)
#        
# # The data is now ready to create a BOW model (sparse matrix)
# from sklearn.feature_extraction.text import CountVectorizer
# cv =CountVectorizer(max_features = 250)      
# X = cv.fit_transform(corpus).toarray()
# 
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# bow_dataset.labels = le.fit_transform(bow_dataset.labels)
# y = bow_dataset.iloc[:,1].values
# 
# X_train = X[:-327,:]
# X_total = X_train
# X_test = X[-327:,:]
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X_total, y, test_size=0.2, random_state=2)
# 
# # Gradient boost classifier
# from sklearn.ensemble import GradientBoostingClassifier
# clf_gb = GradientBoostingClassifier()
# clf_gb.fit(X_train,y_train)
# 
# y_pred = clf_gb.predict(X_val)
# y_pred= le.inverse_transform(y_pred)
# 
# ###############################################################################
# top_category = []
# i=0
# for x in category_tree:
#        if len(x)>1:
#               top_category.append(x[0])
#        else:
#               top_category.append(y_pred[i])
#               i++
# 
# # Now we have all our categories figured out we can drop the category tree column
# # and replace it with top_category column
# dataset = pd.concat([dataset, pd.DataFrame(np.array(top_category))], axis =1)
# dataset=dataset.drop(['product_category_tree'], axis =1)
# =============================================================================
