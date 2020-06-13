# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:23:02 2020

@author: Siddham Jain
"""

from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.layers import Input
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import regularizers
import pandas as pd

##getting the shape of images
#img_path = 'C:/Users/MSI_PC/Desktop/RH intern task/question1/train/1.png'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#print (x.shape)
#x = np.expand_dims(x, axis=0)
#print (x.shape)
#print('Input image shape:', x.shape)


##training data
data_path = 'C:/Users/MSI_PC/Desktop/RH intern task/question1/train'
data_dir_list = os.listdir(data_path)

img_data_list=[]
img_data_name_list = []

for img in data_dir_list:
    img_data_name_list.append(img)
    img_path=data_path+'/'+img
    img=image.load_img(img_path,target_size=(64,64))
    x=image.img_to_array(img)
    img_data_list.append(x)

img_data = np.array(img_data_list)

x_total = img_data
x_total /= 255 # We always need to normalise in DL models

dataset = pd.read_csv(r'C:\Users\MSI_PC\Desktop\RH intern task\question1\Train.csv')
Y = []
y = dataset['category'].values
for i in data_dir_list:
       Y.append(y[int(i[:-4])-1])
Y = np_utils.to_categorical(Y)
Y = np.delete(Y,0,1)  #col 1 corresponds to class 0 so we delete it

X_train, X_val, y_train, y_val = train_test_split(x_total, Y, test_size=0.1, random_state=2)

#### model arctitecture

clf = Sequential()
clf.add(Conv2D(filters = 32,input_shape = (64, 64, 3),
               kernel_size= 3, use_bias= True))

clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

clf.add(Conv2D(16, (3, 3)))
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

clf.add(Flatten())
clf.add(Dense(128, activation= 'relu'))
clf.add(Dense(16, activation= 'softmax'))

clf.summary()

####

clf.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
t=time.time()

hist = clf.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_val, y_val))
hist = hist.history
print('Training time: %s' % (t - time.time()))

# As the train accuracy is -> 98% and validation accuracy ->85% the model is overfitiing
# Hence we'll add a dropout layer
# Preventing Overfitting
clf = Sequential()
clf.add(Conv2D(filters = 32,input_shape = (64, 64, 3),
               kernel_size= 3, use_bias= True))

clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

clf.add(Conv2D(16, (3, 3)))
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2), strides= (2,2)))

clf.add(Flatten())
clf.add(Dense(128, activation= 'relu'))
clf.add(Dropout(.3))
clf.add(Dense(16, activation= 'softmax'))

clf.summary()
hist_improv = clf.fit(X_train, y_train, epochs=15, verbose=1, validation_data=(X_val, y_val))


# to find the optimal no of epochs  #Seems its 6
plt.plot(range(1,16), hist_improv.history['loss'], label = 'Train Loss')
plt.plot(range(1,16), hist_improv.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.xlabel('No of Epochs')
plt.ylabel('Losses')
plt.show()

# The train accuracies ~ 92% val_accuracies ~ 85%  [dropout = .5]

# Lets use regularization.l2(.01) instead of Droupout layer to prevent overfitiing
# only in Dense layer we get an even better model with 20 epochs

###############3

# Computing f1_score
y_pred_val = clf.predict_classes(X_val)
y_pred_val = np_utils.to_categorical(y_pred_val)
from sklearn.metrics import f1_score, accuracy_score
f1 = f1_score(y_val, y_pred_val, average= 'macro')  #.876
acc_val = accuracy_score(y_val, y_pred_val)    # 87.8%



####testing data
data_path_test = 'C:/Users/MSI_PC/Desktop/RH intern task/question1/test/test'
dataset_test = pd.read_csv(r'C:\Users\MSI_PC\Desktop\RH intern task\question1\Test.csv')

img_data_list=[]

for img in dataset_test.name:
    img_path=data_path_test+'/'+img
    img=image.load_img(img_path,target_size=(64,64))
    x=image.img_to_array(img)
    img_data_list.append(x)

X_test = np.array(img_data_list)
X_test /= 255
y_pred = clf.predict_classes(X_test)
for j in range(0,len(y_pred)):
       y_pred[j]+=1

submission = pd.concat([dataset_test, pd.DataFrame(y_pred)], axis = 1)
#we can change column names of pandas dataframe by 
# submission.columns = ['name_1','name_2']

submission.to_csv(path_or_buf = r'C:\Users\MSI_PC\Desktop\RH intern task\question1\predictions_submission.csv', 
                  index =False)



