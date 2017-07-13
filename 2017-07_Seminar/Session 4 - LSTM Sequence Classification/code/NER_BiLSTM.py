# -*- coding: utf-8 -*-
"""
This is a German NER implementation for the dataset GermEval 2014. Also check the folder 'Session 1 - SENNA' for some further information.

This model uses a bi-directional LSTM and runs over an input sentence. The output of the BiLSTM is inputted to a softmax-layer to derive
the probabilities for the different tags.



Single BiLSTM (100 hidden units), after 15 epochs:
Dev-Data: Prec: 0.789, Rec: 0.739, F1: 0.763
Test-Data: Prec: 0.781, Rec: 0.717, F1: 0.747

Stacked BiLSTM (2 layers, 64 hidden states), after 25 epochs:
Dev-Data: Prec: 0.804, Rec: 0.763, F1: 0.783
Test-Data: Prec: 0.795, Rec: 0.738, F1: 0.766


Code was written & tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 and tensorflow 1.2.1
- Keras 2.0.5


@author: Nils Reimers, www.deeplearning4nlp.com
"""
from __future__ import print_function
import numpy as np
import random
import time
import gzip

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: 
    #Python 2.7 imports
    import cPickle as pkl


import BIOF1Validation

import keras
import keras
from keras.models import Model
from keras.layers import *




f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']
caseEmbeddings = embeddings['caseEmbeddings']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('pkl/data.pkl.gz', 'rb')
train_data = pkl.load(f)
dev_data = pkl.load(f)
test_data = pkl.load(f)
f.close()



#####################################
#
# Create the  Network
#
#####################################

n_out = len(label2Idx)

words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)

casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)

output = concatenate([words, casing])
output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(n_out, activation='softmax'))(output)

#Create our model and compile it using Nadam optimizer with categorical cross-entropy for sparse y-labels
model = Model(inputs=[words_input, casing_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()



##################################
#
# Training of the Network
#
##################################

def iterate_minibatches(dataset): 
    endIdx = len(dataset)
    
    for idx in range(endIdx):
        tokens, casing, labels = dataset[idx]        
            
        labels = np.expand_dims([labels], -1)     
        yield labels, np.asarray([tokens]), np.asarray([casing])


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    for tokens, casing, labels in dataset:    
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        pred = model.predict([tokens, casing], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        
        
    return predLabels, correctLabels
        
number_of_epochs = 20

print("%d epochs" % number_of_epochs)

print("%d train sentences" % len(train_data))
print("%d dev sentences" % len(dev_data))
print("%d test sentences" % len(test_data))


for epoch in range(number_of_epochs):    
    print("--------- Epoch %d -----------" % epoch)
    random.shuffle(train_data)
    start_time = time.time()    
    
    #Train one sentence at a time (i.e. online training) to avoid padding of sentences
    cnt = 0
    for batch in iterate_minibatches(train_data):
        labels, tokens, casing = batch       
        model.train_on_batch([tokens, casing], labels)   
        cnt += 1
        
        if cnt % 100 == 0:
            print('Sentence: %d / %d' % (cnt, len(train_data)), end='\r')
    print("%.2f sec for training                 " % (time.time() - start_time))
    
    
    #Performance on dev dataset        
    predLabels, correctLabels = tag_dataset(dev_data)        
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
    print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
    
    #Performance on test dataset       
    predLabels, correctLabels = tag_dataset(test_data)        
    pre_test, rec_test, f1_test= BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
    print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
    
    print("%.2f sec for evaluation" % (time.time() - start_time))
    print("")
        
