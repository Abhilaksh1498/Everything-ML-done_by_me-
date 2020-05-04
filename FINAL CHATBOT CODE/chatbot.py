# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:35:02 2020

@author: MSI_PC
"""
import os
import yaml
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import preprocessing, utils
lines = open(r'C:\Users\MSI_PC\Desktop\Chatbot\Final Chatbot code\cornell movie-dialogs corpus\movie_lines.txt', 
             encoding = 'utf-8', errors = 'ignore').readlines()
conversations = open(r'C:\Users\MSI_PC\Desktop\Chatbot\Final Chatbot code\cornell movie-dialogs corpus\movie_conversations.txt', 
                     encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    id2line[_line[0]] = _line[-1]
    
# there aren't any errors in movie_lines file as
# set([len(x.split(' +++$+++ ')) for x in lines]) = 5
    
# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "")
    conversations_ids.append(_conversation.split(', '))
    
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Loading the kaggle chatterbot dataset
files_path = 'C:\\Users\\MSI_PC\\Desktop\\Chatbot\\Final Chatbot code\\kaggle_chatterbot_data\\'
files_list = os.listdir('C:\\Users\\MSI_PC\\Desktop\\Chatbot\\Final Chatbot code\\kaggle_chatterbot_data\\')
for file_name in files_list:
       stream = open(files_path+ file_name,'rb')
       docs = yaml.safe_load(stream)
       for con in docs['conversations']:
              try:
                     answers.append(' '.join(con[1:]))
                     questions.append(con[0])
              except:
                     continue
                     
def clean_text(text):
       text = text.lower()
       text = re.sub(r"i'm", "i am", text)
       text = re.sub(r"he's", "he is", text)
       text = re.sub(r"she's", "she is", text)
       text = re.sub(r"that's", "that is", text)
       text = re.sub(r"what's", "what is", text)
       text = re.sub(r"where's", "where is", text)
       text = re.sub(r"how's", "how is", text)
       text = re.sub(r"\'ll", " will", text)
       text = re.sub(r"\'ve", " have", text)
       text = re.sub(r"\'re", " are", text)
       text = re.sub(r"\'d", " would", text)
       text = re.sub(r"won't", "will not", text)
       text = re.sub(r"can't", "cannot", text)
       text = re.sub(r"n't", " not", text)
       text = re.sub( '[^a-zA-Z]', ' ', text )
       return text

# while using tokenize punctutations are removed except ' 
# here i've tried to split most common words containing '
for i in range(len(questions)):
       questions[i] = clean_text(questions[i])
       answers[i] = clean_text(answers[i])

# We'll tokenize answers and questions
from keras.preprocessing.text import Tokenizer       
tokenizer = Tokenizer(oov_token= 'UNK', char_level= False)
tokenizer.fit_on_texts(questions + answers)

# That + 1 is because of reserving padding (i.e. index zero)
tokenizer.word_index['<SOS>'] = len(tokenizer.word_index)+1
tokenizer.word_index['<EOS>'] = len(tokenizer.word_index)+1
#from gensim.models import Word2Vec
#
vocab = list(tokenizer.word_index.keys())
# We'll remove too long sentences (either question or answers) as it will take un-necessary
#memory while training
# using counter class we find most of questions/answers <= 75 words
req_ques = []
req_ans = []
for i in range(len(answers)):
       if (len(answers[i].split()) > 75 or len(questions[i].split())>75):
              continue
       else:
              req_ans.append(answers[i])
              req_ques.append(questions[i])
answers = req_ans
questions = req_ques
       

# encoder input data
tokenized_questions = tokenizer.texts_to_sequences( questions )
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions ,                           maxlen=maxlen_questions , padding='post' )
encoder_input_data = np.array( padded_questions )
# print( encoder_input_data.shape , maxlen_questions )

# We need to append <SOS> and <EOS> to answers
# since we are dealing in numbers we'll just add the required token in tokenized answers
# texts_to_sequences cleans text before splitting into words

# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
tokenized_answers = [([tokenizer.word_index['<SOS>']] + tokenized_answers[i] + [tokenizer.word_index['<EOS>']]) for i in range(len(tokenized_answers))]
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
# print( decoder_input_data.shape , maxlen_answers )

VOCAB_SIZE = len(tokenizer.word_index) +1

# decoder_output_data
for i in range(len(tokenized_answers)) :
       tokenized_answers[i] = tokenized_answers[i][1:]
# We are removing <SOS> token
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = np.zeros((padded_answers.shape[0], padded_answers.shape[1], VOCAB_SIZE))
onehot_answers = utils.to_categorical(padded_answers , VOCAB_SIZE)
decoder_output_data = np.array( onehot_answers )
print(decoder_output_data.shape)