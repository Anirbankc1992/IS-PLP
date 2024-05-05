# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:16:16 2019

@author: isswan
"""

#For windows Please install git from https://git-scm.com/download/win
#Then install keras-contrib as below
#pip install git+https://www.github.com/keras-team/keras-contrib.git

import keras
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import en_core_web_sm



##prepare dataset
from os import getcwd, chdir
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 
import en_core_web_sm
nlp = en_core_web_sm.load()

file_location = fpath+"//data//train_AddToPlaylist_full.json"
test_file = fpath+"//data//validate_AddToPlaylist.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
test_file_stream = codecs.open(test_file, 'r', 'utf-8')
jdata = json.load(file_stream)
jtestdata = json.load(test_file_stream)

def loadData(jdata):
    data_List = []
    vocab = []
    tagVocab=[]
    
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
                        
        vocab.extend(wordList)
        tagVocab.extend(tagList)
        
        sent = ' '.join(wordList)
        sent_nlp = nlp(sent)
        
        for token in sent_nlp:
            posList.append(token.tag_)
    
        for idx,word in enumerate(wordList):
            sentlist.append((word,posList[idx],tagList[idx]))
    
        data_List.append(sentlist)
        
    return data_List,list(set(vocab)),list(set(tagVocab))

train_list, vocab, tagVocab = loadData(jdata)
test_list, testvocab, testagVocab = loadData(jtestdata)

print (len(train_list))
print (len(test_list))
len(testagVocab)
len(tagVocab)

##For sentence padding and UNKnown words
vocab.append('ENDPAD')
vocab.append('UNK')

########################################################
###index the data

word2idx = {w: i + 1 for i, w in enumerate(vocab)}
word2idx["Winston"]
word2idx['UNK']
word2idx['ENDPAD']

idx2word = {value: key for key, value in word2idx.items()}

###check the index
for i, item in enumerate(word2idx.items()):
    print(item)
    if i > 5:
        break

###check the Nth word 
for key, value in word2idx.items():
    if value == 2936:
        print(key)

X_train = [[word2idx[w[0]] for w in s] for s in train_list]
train_list[0]
X_train[0]

####Index X_test using training vocab
###X_test = [[word2idx[w[0]] for w in s] for s in test_list] is not working due to UNK

X_test=[]
for s in test_list:
    sent_index=[]
    for w in s:
        if w[0] in word2idx:
            sent_index.append(word2idx[w[0]])
        else:
            sent_index.append(word2idx['UNK'])
    X_test.append(sent_index)
    
test_list[0]
X_test[0]

##index the taglist with longger one 
tag2idx = {t: i for i, t in enumerate(testagVocab)} 


idx2tag = {value: key for key, value in tag2idx.items()}
####################################################
##Pad the sentence
from keras.preprocessing.sequence import pad_sequences
max_len = 32

##Index word in sentences
X_train = pad_sequences(maxlen=max_len, sequences=X_train, padding="post", value=word2idx['ENDPAD'])
X_train[0]

X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=word2idx['ENDPAD'])
X_test[0]

##Index tag sequence.
Y_train = [[tag2idx[w[2]] for w in s] for s in train_list]
Y_train [0]

Y_train = pad_sequences(maxlen=max_len, sequences=Y_train, padding="post", value=tag2idx["O"])
Y_train[0]

print(tag2idx)

Y_test = [[tag2idx[w[2]] for w in s] for s in test_list]
Y_test [0]

Y_test = pad_sequences(maxlen=max_len, sequences=Y_test, padding="post", value=tag2idx["O"])
Y_test[0]

##change the labels y to categorial.

from keras.utils import to_categorical
Y_train = [to_categorical(i, num_classes=len(testagVocab)) for i in Y_train]
Y_train[0]


Y_test = [to_categorical(i, num_classes=len(testagVocab)) for i in Y_test]
Y_test[0]

####################################################
##Setup the CRF-LSTM
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF



input = Input(shape=(max_len,))

print (input)

model = Embedding(input_dim=len(vocab) + 1, output_dim=20,
                  input_length=max_len)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="tanh"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(len(testagVocab))  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

##############################################################################
#Train the model

history = model.fit(X_train,np.array(Y_train), batch_size=32, epochs=20, validation_data=(X_test,np.array(Y_test)), verbose=1)


hist = pd.DataFrame(history.history)

print (hist)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["val_crf_viterbi_accuracy"])
plt.show()

##################################################
##ACC is not enough 
###################################################
predictions = model.predict(X_test)

#################################################
#try with the first prediction

predictions[0]
p0 = np.argmax(predictions[0], -1)
p0

true0 = np.argmax(Y_test[0], -1)
true0

' '.join([idx2word[inx] for inx in X_test[0] if inx!=word2idx['ENDPAD']])

' '.join([idx2tag[inx] for inx in p0])



print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")

len(X_test[0])

for w, t, pred in zip(X_test[0], true0, p0):
    if w != 0:
        print("{:15}: {:5} {}".format(vocab[w-1], testagVocab[t], testagVocab[pred]))

##################################################################################
#evaluate the whole test set

def indexToTag(predictions):
    predList=[]
    for pred in predictions:
        tagList=[]
        for index in np.argmax(pred, -1):
            tagList.append(idx2tag [index])
        predList.append(tagList)    
    return predList
    
predList=indexToTag(predictions)
trueList=indexToTag(Y_test)
        


from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

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

print(bio_classification_report(trueList, predList))










