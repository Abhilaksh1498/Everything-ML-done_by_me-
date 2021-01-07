# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:59:37 2020

@author: MSI_PC
"""

# when you call an nlp object on a string
# spaCy first uses a tokenizer creating a Doc object and then runs all the pipeline components in order

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Print the names of the pipeline components in the current nlp object
print(nlp.pipe_names)

# Print the full pipeline of (name, component) tuples
print(nlp.pipeline) 

# both return a list

##### Adding custom components in pipeline
# Custom components are great for adding custom values to documents, tokens and spans, and customizing the doc.ents
# they cant however add support for additional language (because the nlp model is already instantiated with lang)
# also they cant change weights of pre-trained models like pos tagger etc.
def custom_component(doc):  # take in doc and return a doc
       print(len(doc))
       return doc
nlp.add_pipe(custom_component, first = True)
# other possible arguments are last = True, before = 'ner', after = ...
doc = nlp('some sample text')

########### A helpful example summing up all of the above ############
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
nlp = spacy.load("en_core_web_sm")
animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animals))
print("animal_patterns:", animal_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", None, *animal_patterns)
​
# Define the custom component
def animal_component(doc):
    # Apply the matcher to the doc
    matches = matcher(doc)
    # Create a Span for each match and assign the label "ANIMAL"
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
    # Overwrite the doc.ents with the matched spans
    doc.ents = list(doc.ents) + spans
    return doc
​
​
# Add the component to the pipeline after the "ner" component
nlp.add_pipe(animal_component, after="ner")
print(nlp.pipe_names)
​
# Process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])

############## Scaling & Performance ###############
# if we only want to tokenize text and need a doc object we can use
only_tokenize = nlp.make_doc(TEXT)

# to disable some pipeline components
# Disable the tagger and parser (temporarily i.e. only inside this WITH block)
with nlp.disable_pipes("tagger", "parser"):
    # Process the text
    doc = nlp(text)
    
# To make several docs out of a list of text use nlp.pipe(list of TEXT) => generator and yeilds (doc objects, context)

# Register the Doc extension "author" (default None)
Doc.set_extension("author", default=None)

# Register the Doc extension "book" (default None)
Doc.set_extension("book", default=None)

for doc, context in nlp.pipe(DATA, as_tuples=True):
    # here data was of the form (text,context)
    doc._.book = context["book"]
    doc._.author = context["author"]
