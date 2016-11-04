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
- Python 2.7
- Theano 0.8.2
- Keras 1.1.0


@author: Nils Reimers
"""
import numpy as np
import theano
import theano.tensor as T
import random


import time
import gzip

import GermEvalReader
import BIOF1Validation

import keras
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from docutils.languages.af import labels




trainFile = '../../Session 1 - SENNA/code/data/NER-de-train.tsv'
devFile = '../../Session 1 - SENNA/code/data/NER-de-dev.tsv'
testFile = '../../Session 1 - SENNA/code/data/NER-de-test.tsv'



print "NER with Keras, float: %s" % (theano.config.floatX)


#####################
#
# Read in the vocab
#
#####################
print "Read in the vocab"
vocabPath =  '../../Session 1 - SENNA/code/embeddings/GermEval.vocab.gz'

word2Idx = {} #Maps a word to the index in the embeddings matrix
embeddings = [] #Embeddings matrix

with gzip.open(vocabPath, 'r') as fIn:
    idx = 0               
    for line in fIn:
        split = line.strip().split(' ')                
        embeddings.append(np.array([float(num) for num in split[1:]]))
        word2Idx[split[0]] = idx
        idx += 1
        
embeddings = np.asarray(embeddings, dtype='float32')
embedding_size = embeddings.shape[1]


        
# Create a mapping for our labels
label2Idx = {'O':0}
idx = 1

for bioTag in ['B-', 'I-']:
    for nerClass in ['PER', 'LOC', 'ORG', 'OTH']:
        for subtype in ['', 'deriv', 'part']:
            label2Idx[bioTag+nerClass+subtype] = idx 
            idx += 1
            
#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}


#Casing matrix
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
caseMatrix = np.identity(len(caseLookup), dtype=theano.config.floatX)
            
     
# Read in data   
print "Read in data and create matrices"    
train_sentences = GermEvalReader.readFile(trainFile)
dev_sentences = GermEvalReader.readFile(devFile)
test_sentences = GermEvalReader.readFile(testFile)

# Create numpy arrays
train_data= GermEvalReader.createDataset(train_sentences, word2Idx, label2Idx, caseLookup)
dev_data = GermEvalReader.createDataset(dev_sentences, word2Idx, label2Idx, caseLookup)
test_data = GermEvalReader.createDataset(test_sentences, word2Idx, label2Idx, caseLookup)



#####################################
#
# Create the  Network
#
#####################################


n_out = len(label2Idx)
caseMatrix = np.identity(len(caseLookup), dtype=theano.config.floatX)

        
print "Embeddings shape: ",embeddings.shape

tokens = Sequential()
tokens.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],  weights=[embeddings], trainable=False))

casing = Sequential()
casing.add(Embedding(output_dim=caseMatrix.shape[1], input_dim=caseMatrix.shape[0], weights=[caseMatrix], trainable=False)) 
  
model = Sequential();
model.add(Merge([tokens, casing], mode='concat'))  
model.add(Bidirectional(LSTM(100, return_sequences=True, dropout_W=0.2))) 
model.add(TimeDistributedDense(n_out, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-7, momentum=0.0, nesterov=False, clipvalue=3) 
rmsprop = RMSprop(clipvalue=3) 
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.summary()


##################################
#
# Training of the Network
#
##################################

def iterate_minibatches(dataset, startIdx, endIdx): 
    endIdx = min(len(dataset), endIdx)
    
    for idx in xrange(startIdx, endIdx):
        tokens, casing, labels = dataset[idx]             
        labels = np_utils.to_categorical(labels, n_out)  
        yield np.asarray([labels]), np.asarray([tokens]), np.asarray([casing])


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    for tokens, casing, labels in dataset:    
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        pred = model.predict_classes([tokens, casing], verbose=False)[0]               
        correctLabels.append(labels)
        predLabels.append(pred)
        
        
    return predLabels, correctLabels
        
number_of_epochs = 20
stepsize = 24000
print "%d epochs" % number_of_epochs

print "%d train sentences" % len(train_data)
print "%d dev sentences" % len(dev_data)
print "%d test sentences" % len(test_data)

for epoch in xrange(number_of_epochs):    
    print "--------- Epoch %d -----------" % epoch
    random.shuffle(train_data)
    for startIdx in xrange(0, len(train_data), stepsize):
        start_time = time.time()    
        for batch in iterate_minibatches(train_data, startIdx, startIdx+stepsize):
            labels, tokens, casing = batch       
            model.train_on_batch([tokens, casing], labels)   
        print "%.2f sec for training" % (time.time() - start_time)
        
        
        #Train Dataset       
        start_time = time.time()    
        predLabels, correctLabels = tag_dataset(train_data)        
        pre_train, rec_train, f1_train = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        print "Train-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_train, rec_train, f1_train)
        
        #Dev Dataset       
        predLabels, correctLabels = tag_dataset(dev_data)        
        pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        print "Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev)
        
        #Test Dataset       
        predLabels, correctLabels = tag_dataset(test_data)        
        pre_test, rec_test, f1_test= BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        print "Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test)
        
        print "%.2f sec for evaluation" % (time.time() - start_time)
        print ""
        
