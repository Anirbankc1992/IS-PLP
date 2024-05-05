# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:35:59 2019

@author: isswan
"""
#pip install nltk
#pip install spacy
#python -m spacy download en_core_web_sm
#python -m spacy download en_core_web_md
#python -m spacy download en_core_web_lg

import os
import json
import codecs
import pandas as pd
import numpy as np
import nltk
from nltk.tag import hmm

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize, pos_tag, ne_chunk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import en_core_web_md
#import en_core_web_lg

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

##prepare dataset
from os import getcwd, chdir
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 



file_location = fpath+"//data//train_AddToPlaylist_full.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

# append state  <S> <\S> with label O 
# generate state labels B_Entity I_Entity 

train_list = []
for data in jdata['AddToPlaylist']:
    sentlist=[]
    sentlist.append(('<S>','O'))
    for sequence in data['data']:
        if 'entity' not in sequence:
            tokenList = sequence['text'].lower().split()
            for tok in tokenList:
                sentlist.append((tok,'O'))
        else:
            tokenList = sequence['text'].lower().split()
            for idx,tok in enumerate(tokenList):
                if idx:
                    sentlist.append((tok,'I_'+sequence['entity']))
                else:
                    sentlist.append((tok,'B_'+sequence['entity']))  
    sentlist.append(('<\S>','O'))
    train_list.append(sentlist)
    
print (len(train_list))
##Explore some training data

##############################

##########################################################
# Import HMM module
# And train with the data

trainer = hmm.HiddenMarkovModelTrainer()

tagger = trainer.train_supervised(train_list)

print (tagger)


test = "add this track by Kevin Cadogan to my playlist"
print (tagger.tag(test.split()))

test = "add Caleigh Peters to my women of country list"
print (tagger.tag(test.split()))

## Person Names are difficult to recognize 
## delexiconlization is needed
##########################################################
## Get help from Person Name Recognizer from NLTK 

nlp = en_core_web_sm.load()
snowball = nltk.SnowballStemmer('english')


doc = nlp('add this track by Kevin Cadogan to my playlist')
print([(X.text,X.label_) for X in doc.ents])

##Target : add this track by NE_PERSON to my playlist
def replaceNE(NE,sent):
    doc = nlp(sent)
    for ent in doc.ents:
        if ent.label_==NE:
            sent = sent.replace(ent.text,"NE_"+ent.label_)
    return sent

print(replaceNE('PERSON','add this track by Kevin Cadogan to my playlist'))

def preprocess(text):
    text = replaceNE('PERSON',text)
    text = snowball.stem(text)
    return text.lower()

train_list_NE = []

for data in jdata['AddToPlaylist']:
    sentlist=[]
    sentlist.append(('<S>','O'))
    
    for sequence in data['data']:
        
        text = sequence['text']
        text = preprocess(text)
        tokenList = text.split()
        
        if 'entity' not in sequence:
            for tok in tokenList:
                sentlist.append((tok,'O'))
        else:
            for idx,tok in enumerate(tokenList):
                if idx:
                    sentlist.append((tok,'I_'+sequence['entity']))
                else:
                    sentlist.append((tok,'B_'+sequence['entity']))  
                    
    sentlist.append(('<\S>','O'))
    train_list_NE.append(sentlist)
    
print (len(train_list_NE))

###########################################################
tagger_NE = trainer.train_supervised(train_list_NE)

print (tagger_NE)

test = "add Caleigh Peters to my women of country list"

print (preprocess(test))

print (tagger_NE.tag(preprocess(test).split()))

##########################################################
###Evaluation to be finished
##########################################################
test_list_NE = []
test_labels= []

file_location = fpath+"//data//validate_AddToPlaylist.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

for data in jdata['AddToPlaylist']:
    labellist=[]
    sentlist=[]
    for sequence in data['data']:
        
        text = sequence['text']
        text = preprocess(text)        
        tokenList = text.split()
        
        if 'entity' not in sequence:
            tokenList = sequence['text'].lower().split()
            for tok in tokenList:
                sentlist.append(tok)
                labellist.append('O')
        else:
            tokenList = sequence['text'].lower().split()
            for idx,tok in enumerate(tokenList):
                if idx:
                    sentlist.append(tok)
                    labellist.append('I_'+sequence['entity'])
                else:
                    sentlist.append(tok)
                    labellist.append('B_'+sequence['entity'])
                    
    test_list_NE.append(" ".join(sentlist))
    test_labels.append(labellist)
    
print (len(test_list_NE))
print (len(test_labels))

print (test_list_NE[1])
print (test_labels[1])
##################################################
##Predict on top of the test data
##Compare the predicted Labels againest the correct labels



