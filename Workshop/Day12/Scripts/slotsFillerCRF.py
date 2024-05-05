# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:06:04 2019

@author: isswan
"""

import json
import codecs
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pandas as pd

#pip install python-crfsuite

import pycrfsuite
import spacy
import en_core_web_sm

print(sklearn.__version__)
##########################################################


##prepare dataset
from os import getcwd, chdir
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 
import en_core_web_sm
nlp = en_core_web_sm.load()

file_location = fpath+"//data//train_AddToPlaylist_full.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

def loadData(jdata):
    data_List = []
    
    for data in jdata['AddToPlaylist']:
        
        wordList=[]
        tagList=[]
        posList=[]
        sentlist=[]
        
        for sequence in data['data']:   
            text = sequence['text']
            tokenList = text.split()
            
            if 'entity' not in sequence:
                for tok in tokenList:
                    wordList.append(tok)
                    tagList.append('O')
            else:
                for idx,tok in enumerate(tokenList):
                    wordList.append(tok)
                    if idx:
                        tagList.append('I-'+sequence['entity'])
                    else:
                        tagList.append('B-'+sequence['entity'])
                        
        sent = ' '.join(wordList)
        sent_nlp = nlp(sent)
        
        for token in sent_nlp:
            posList.append(token.tag_)
    
        for idx,word in enumerate(wordList):
            sentlist.append((word,posList[idx],tagList[idx]))
    
        data_List.append(sentlist)
    return data_List

train_list = loadData(jdata)

print (len(train_list))

###############################################################
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [  # for all words
        'bias',
        'word.lower=' + word.lower(),
        #'word[-3:]=' + word[-3:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0: # if not <S>
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')  # beginning of statement
        
    if i < len(sent)-1:  # if not <\S>
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
#################################################################

df_1 = pd.DataFrame(train_list[1],columns=["Word","POS","Entity or Aspect Tag"])
# change to dataframe for easy printing.
df_1

features = sent2features(train_list[1])
###############################################################
X_train = [sent2features(s) for s in train_list]
y_train = [sent2labels(s) for s in train_list]


file_location = fpath+"//data//validate_AddToPlaylist.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

test_list = loadData(jdata)

X_test = [sent2features(s) for s in test_list]
y_test = [sent2labels(s) for s in test_list]
################################################################
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.params()

trainer.train('addPlayList.crfsuite')
#####################################################################
tagger = pycrfsuite.Tagger()
tagger.open('addPlayList.crfsuite')

example_sent = X_test[0]

print("Predicted:[\'", '\', \''.join(tagger.tag(example_sent))+'\']')
print("Correct:  ", y_test[0])

###################################################################
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

y_pred = [tagger.tag(xseq) for xseq in X_test]

print(bio_classification_report(y_test, y_pred))

##############################################################
##Let's check what classifier learned

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])

##Check the state features:
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])

