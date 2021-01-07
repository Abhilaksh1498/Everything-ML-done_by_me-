# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:03:55 2020

@author: MSI_PC
"""

import spacy

# load english language
from spacy.lang.en import English
nlp = English()

some_text = 'abra cadbra 69 $'
doc = nlp(text)

# printing text of doc
print(doc.text)

# doc objects contains a sequence of tokens and is iterable and splicable just like list
first_word = doc[0]
print(first_word.text)

first_three_words = doc[:-1]   # abra cadabra 69, note that 69 is treated as a single number just as it should be
for token in doc:
       print(f'Index {token.i}')
       print(f'Word {token.text}')   

bool1 = doc[0].is_alpha # true
bool2 = doc[2].like_num  # true
nlp('has a % sign and some ! punctuation')[2].is_punct  # true

# to see a list of languages and their model support visit https://spacy.io/usage/models
# to download a package use python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_sm  --> NEEDS TO BE DONE IN ANACONDA PROMPT
nlp = spacy.load('en_core_web_sm')
# its an nlp object like the previous one all all those prev attributes

############# Predictiong linguistic annotations (pos tagging, dependency label, named entities) #############

text = "It's official: Apple is the first U.S. public company to reach a $1 trillion market value"
​doc = nlp(text)
for token in doc:
       print(f'Text is {token.text} Pos token {token.pos_} and dependency label is {token.dep_}')
       
#named entities can be found using .ents attribute and its category using .label_ attribute
for ent in doc.ents:
    # Print the entity text and label
    print(ent.text, ent.label_)

# the entity names, POS predicted may not be clear, so we can use spacy.explain()
# spacy.explain("ORG") 


########### Using matcher ########## 
# Import the Matcher
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp("Upcoming iPhone X release date leaked as Apple reveals pre-orders")

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern matching two tokens: "iPhone" and "X"
pattern1 = [{"TEXT" : 'iPhone'},{"TEXT":'X'}]
pattern2 = [{'LOWER' : 'apple'},{"TEXT":'pre-orders'}] 
# Add the pattern to the matcher
matcher.add("IPHONE_X_PATTERN", None, pattern1)
matcher.add("APPLE_PATTERN", None, pattern2)
# Use the matcher on the doc
matches = matcher(doc)     # returns a list of 3d tuples
print("Matches:", [doc[start:end].text for match_id, start, end in matches])

# Write a pattern that matches a form of "download" plus proper noun
pattern = [{"LEMMA": 'download'}, {"POS": "PROPN"}]