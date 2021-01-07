# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:23:14 2020

@author: MSI_PC
"""

import spacy

# nlp.vocab.strings is a 2 way dictionary 
# nlp could either be a model or the naive English() class

nlp = spacy.load("en_core_web_sm")
doc = nlp("David Bowie is a PERSON")

# Look up the hash for the string label "PERSON"
person_hash = nlp.vocab.strings["PERSON"]
print(person_hash)

# Look up the person_hash to get the string
person_string = nlp.vocab.strings[person_hash]
print(person_string)

########## as you might observe nlp object is universal (not dependent on text), its the doc object that carries
# info about our particular use case
# so, hashes generated are universal, here even if we searched for hash of a very common english word "ball" it
# returns a value even though its not in doc text
# any new words encountered will be added to dictionary when we try to lookup for their hash

#### creating Doc and Span objects from built in classes
from spacy.tokens import Doc, Span
# suppose the text is same as in previous example
words = ["David", "Bowie", "is",'a','PERSON','!']
spaces_after_all_tokens = [True, True, True,True, False, False]

doc = Doc(nlp.vocab, words, spaces_after_all_tokens)
print(doc.text) 

span = Span(doc, 0,2,label = "PERSON")
print(span.text, span.label_)

type(doc.ents)
doc.ents = [span]
# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])


############# Spacy's word vectors (300d) and similarities ##############
# you'll need large or medium models for this
# Load the en_core_web_md model
nlp = spacy.load('en_core_web_md')

# Process a text
doc = nlp("Two bananas in pyjamas")

# Get the vector for the token "bananas"
bananas_vector = doc[1].vector
print(bananas_vector)

# we also have vectors for span and doc type objects, their value is average of token vectors in them
doc_vector = doc.vector
in_pyjamas_vector = doc[-2:].vector 

# We can also compare similarity using similairty() method AMONG doc, token, span objects (even diff object types)
similarity = doc2.similarity(doc1)  #cosine similarity by default

# you can use Phrase Matcher to match text directly instead of specifying attributes of tokens
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
matcher.add("COUNTRY", None, *patterns)
for match_id, start, end in matcher(doc):
    # Create a Span with the label for "GPE"
    span = Span(doc, start, end, label="GPE")

    # Overwrite the doc.ents and add the span
    doc.ents = list(doc.ents) + [span]

    # Get the span's root head token
    span_root_head = span.root.head
    # Print the text of the span root's head token and the span text
    print(span_root_head.text, "-->", span.text)

# Print the entities in the document
print([(ent.text, ent.label_) for ent in doc.ents])
