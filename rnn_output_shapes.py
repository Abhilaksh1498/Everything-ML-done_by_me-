# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:47:22 2020

@author: MSI_PC
"""

import keras
from keras.models import Model
from keras.layers import GRU, LSTM, Input

import numpy as np
import random 

T = 3  # mimicing seq len
D =  5  # Mimicing word embeddings dimension
M = 10   # num_rnn_units
X = np.random.randn(1,T,D)

input_ = Input(shape = (T,D))
lstm1 = GRU(M, return_sequences= True)(input_)
lstm_state_no_seq = Model(inputs= input_, outputs = lstm1)
o = lstm_state_no_seq.predict(X)

####################### LEARNING ######################
#1. Both GRU/LSTM h is the last output, if return_seq is false o,h are same
#2. The output at each time step is of dim 1*M (M is rnn_units)
#3. The return_seq = True provides T diff outputs but only 1 h and 1 c (which are the last values)
#4. c,h are always array of dim 1xM whereas o can be of dim 1xTxM (return_seq = True) & of dim 1XM if False
#5. GRU has 2 outputs which are same if return_seq = False
#6. If return_state = False none of h or c will be returned only output of dim relevant to return_seq is outputted


# BIDIRECTIONAL
from keras.layers import Bidirectional
bi1 = Bidirectional(GRU(M, return_state= True))(input_)
bi_gru = Model(inputs= input_, outputs = bi1)
o,h1,h2 = bi_gru.predict(X)

############################ LEARNING ###############################
#1. 2 layers are run simulatenously
#2. First layer in normal order, in 2nd the input is fed in reverse order
#3. h returned are hT of forward layer and h1 of backward layer (i.e. both the h that are calculated at the last) and the output by default is a concatenation of both of these h's (dim of output is 1x2M)
#4. The outputs are in order o,h1,c1,h2,c2 for lstm and o,h1,h2 for GRU
from keras.models import Sequential
model = Sequential()
model.add(Dense(input = ))