# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:36:45 2019

@author: isswan
"""

from __future__ import unicode_literals
import spacy
import numpy as np
print (spacy.__version__)

##prepare dataset
from os import getcwd, chdir
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 

######################################################
##Document/Question Pair Similarity by 

# u need to run for the spacy models
# python -m spacy download en_core_web_sm
'''
English multi-task CNN trained on OntoNotes. 
Assigns context-specific token vectors (not "real" wordvec from Glove)
'''
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg
'''
English multi-task CNN trained on OntoNotes, 
with GloVe vectors trained on Common Crawl.
doc vector = average of word tokens vector
'''


nlp = spacy.load('en_core_web_sm')
tokens = nlp("Apples apple")

for token in tokens:
    print(token.text, token.has_vector,token.vector, token.is_oov)

doc1 = nlp(u"Apples and orange are similar.")
doc2 = nlp(u"Hippos and lions are not.")
doc2 = nlp(u"English multi-task CNN trained on OntoNotes")


    
doc1.similarity(doc2)

#by using the similarity between questions, pick up the top N as the relavent questions and fetch the answers

#########################################################
#Git installation
#pip install deeppavlov
#python -m deeppavlov install tfidf_logreg_en_faq
##Answer FAQ by deeppavlov
'''
tfidf_logreg_en_faq is the model’s configuration file loccated below (windowsOS)
C:\ProgramData\Anaconda3\Lib\site-packages\deeppavlov\configs\faq
The config file consists of four main sections:
    dataset_reader, dataset_iterator, chainer, and train. 
    dataset_reader defines the dataset’s location (url),format
    For this configuration, the similarity is calculated based on tfidf and wordvectors from Glove
    The classifier is logistic regression
'''
from deeppavlov import configs, train_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model


##use local training data and retrain the model

model_config = read_json(configs.faq.tfidf_logreg_en_faq)
model_config["dataset_reader"]["data_path"] = fpath+"\\Data\\music_train_file.csv"
#must None as URL has the highest priority 
model_config["dataset_reader"]["data_url"] = None 

faq = train_model(model_config)
print(faq(["Why is music theory important"]))




