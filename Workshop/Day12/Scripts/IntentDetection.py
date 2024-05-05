# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:38:34 2019

@author: isswan
"""
#conda install sklearn
#pip install pandas
#pip install tensorflow==1.15


import os
import json
import codecs
import pandas as pd
import numpy as np
import random

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics


##prepare dataset
from os import getcwd, chdir
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 

train_list = []

file_location = fpath+"//data//train_AddToPlaylist_full.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

print(jdata['AddToPlaylist'][1])


for data in jdata['AddToPlaylist']:
    line=""
    for sequence in data['data']:
        line += sequence['text']
    train_list.append([line.lower(),0])
print (len(train_list))


file_location = fpath+"//data//train_PlayMusic_full.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

for data in jdata['PlayMusic']:
    line=""
    for sequence in data['data']:
        line += sequence['text']
    train_list.append([line.lower(),1])
print (len(train_list))

file_location = fpath+"//data//train_SearchCreativeWork_full.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

for data in jdata['SearchCreativeWork']:
    line=""
    for sequence in data['data']:
        line += sequence['text']
    train_list.append([line.lower(),2])
print (len(train_list))

######################################################
test_list = []

file_location = fpath+"//data//validate_AddToPlaylist.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

for data in jdata['AddToPlaylist']:
    line=""
    for sequence in data['data']:
        line += sequence['text']
    test_list.append([line.lower(),0])
print (len(test_list))


file_location = fpath+"//data//validate_PlayMusic.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

for data in jdata['PlayMusic']:
    line=""
    for sequence in data['data']:
        line += sequence['text']
    test_list.append([line.lower(),1])
print (len(test_list))

file_location = fpath+"//data//validate_SearchCreativeWork.json"
file_stream = codecs.open(file_location, 'r', 'utf-8')
jdata = json.load(file_stream)

for data in jdata['SearchCreativeWork']:
    line=""
    for sequence in data['data']:
        line += sequence['text']
    test_list.append([line.lower(),2])
print (len(test_list))
########################################################

random.seed(4)
random.shuffle(train_list)
random.shuffle(test_list)


X_train = [t[0] for t in train_list]
X_test = [t[0] for t in test_list]

Y_train = [t[1] for t in train_list]
Y_test = [t[1] for t in test_list]

bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
train_bigram_vectors = bigram_vectorizer.fit_transform(X_train)
test_bigram_vectors = bigram_vectorizer.transform(X_test)

train_bigram_vectors.shape
test_bigram_vectors.shape

ch21 = SelectKBest(chi2, k=1000)
train_bigram_Kbest = ch21.fit_transform(train_bigram_vectors, Y_train)
test_bigram_Kbest = ch21.transform(test_bigram_vectors)

######################################################################
clf_ME = LogisticRegression(random_state=0, solver='lbfgs').fit(train_bigram_Kbest, Y_train)
predME = clf_ME.predict(test_bigram_Kbest)
pred = list(predME)
print(metrics.confusion_matrix(Y_test, pred))
print(metrics.classification_report(Y_test, pred))

model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf')
clr_svm = model_svm.fit(train_bigram_Kbest, Y_train)   
predicted = clr_svm.predict(test_bigram_Kbest)
print(metrics.confusion_matrix(Y_test, predicted))
print(np.mean(predicted == Y_test) )
print(metrics.classification_report(Y_test, predicted))
#####################################################################
##Add FAQ dataset
#####################################################################
file_location = fpath+"//data//MusicFAQ.csv"
faqs=pd.read_table(file_location,header=None,names = ["Label", "Text"])
faqs.head()
faqs.groupby('Label').describe()

questions=faqs[(faqs.Label=="Question")]
questions.head()

print (len(questions))

answers=faqs[(faqs.Label=="Answer")]
print (len(answers))


#Saved for future use
merged_list = tuple(zip(questions['Text'], answers['Text']))
import csv
with open(fpath+"//data//music_train_file.csv", mode='w',newline='',encoding='utf-8') as result_file:
    csv_writer=csv.writer(result_file)
    csv_writer.writerow(['Question','Answer'])
    for row in merged_list:
        csv_writer.writerow(row)



for line in questions['Text']:
    train_list.append([line.lower(),3])
print (len(train_list))


file_location = fpath+"//data//FAQ_Evaluate.txt"
f = open(file_location, "r")

for line in f:
     test_list.append([line.lower(),3])
print (len(test_list))
#######################################################################
##Redo pre-processing

random.seed(4)
random.shuffle(train_list)
random.shuffle(test_list)

X_train = [t[0] for t in train_list]
X_test = [t[0] for t in test_list]

Y_train = [t[1] for t in train_list]
Y_test = [t[1] for t in test_list]

bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
train_bigram_vectors = bigram_vectorizer.fit_transform(X_train)
test_bigram_vectors = bigram_vectorizer.transform(X_test)

train_bigram_vectors.shape
test_bigram_vectors.shape

ch21 = SelectKBest(chi2, k=1000)
train_bigram_Kbest = ch21.fit_transform(train_bigram_vectors, Y_train)
test_bigram_Kbest = ch21.transform(test_bigram_vectors)

train_bigram_Kbest.shape
test_bigram_Kbest.shape

########################################################################

clf_ME = LogisticRegression(random_state=0, solver='lbfgs').fit(train_bigram_Kbest, Y_train)
predME = clf_ME.predict(test_bigram_Kbest)
pred = list(predME)
print(metrics.confusion_matrix(Y_test, pred))
print(metrics.classification_report(Y_test, pred))

model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf')
clr_svm = model_svm.fit(train_bigram_Kbest, Y_train)   
predicted = clr_svm.predict(test_bigram_Kbest)
print(metrics.confusion_matrix(Y_test, predicted))
print(np.mean(predicted == Y_test) )
print(metrics.classification_report(Y_test, predicted))
########################################################################
#########AttnBILSTM
########################################################################
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from AttnBiLSTM.utils.prepare_data import *
import time
from AttnBiLSTM.utils.model_helper import *
import tensorflow as tf
from AttnBiLSTM.attn_bi_lstm import *



# data preprocessing
x_train, x_test, vocab_size = data_preprocessing_v2(X_train, X_test, max_len=32)
print("train size: ", len(x_train))
print("vocab size: ", vocab_size)

# split dataset to test and dev
#x_test, x_dev, y_test, y_dev, dev_size, test_size = split_dataset(x_test, y_test, 0.1)
#print("Validation Size: ", dev_size)

config = {
    "max_len": 32,
    "hidden_size": 64,
    "vocab_size": vocab_size,
    "embedding_size": 128,
    "n_class": 4,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "train_epoch": 5
}

classifier = ABLSTM(config)
classifier.build_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#dev_batch = (x_dev, y_dev)
start = time.time()

for e in range(config["train_epoch"]):

    t0 = time.time()
    print("Epoch %d start !" % (e + 1))
    for x_batch, y_batch in fill_feed_dict(x_train, Y_train, config["batch_size"]):
        return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
        attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
        # plot the attention weight
        # print(np.reshape(attn, (config["batch_size"], config["max_len"])))
    t1 = time.time()

    print("Train Epoch time:  %.3f s" % (t1 - t0))
    #dev_acc = run_eval_step(classifier, sess, dev_batch)
    #print("validation accuracy: %.3f " % dev_acc)

print("Training finished, time consumed : ", time.time() - start, " s")
print("Start evaluating:  \n")

#cnt = 0
#test_acc = 0
#for x_batch, y_batch in fill_feed_dict(x_test, Y_test, config["batch_size"]):
#    acc = run_eval_step(classifier, sess, (x_batch, y_batch))
#    test_acc += acc
#    cnt += 1
#print("Test accuracy : %f %%" % (test_acc / cnt * 100))

feed_dict = make_test_feed_dict(classifier, (x_test, Y_test))
prediction = sess.run(classifier.prediction, feed_dict)

print(metrics.confusion_matrix(Y_test, prediction))
print(np.mean(prediction == Y_test) )
print(metrics.classification_report(Y_test, prediction))





