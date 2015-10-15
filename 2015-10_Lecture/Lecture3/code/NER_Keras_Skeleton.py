# -*- coding: utf-8 -*-
"""
This is a skeleton for you to implement your NER classifier.

You can either choose Lasange (https://github.com/Lasagne/Lasagne) or Keras (http://keras.io)


This file is a skeleton for Keras

@author: Nils Reimers
"""
import numpy as np
import theano
import theano.tensor as T


import time
import gzip

import GermEvalReader
import BIOF1Validation

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding

from KerasLayer.FixedEmbedding import FixedEmbedding


windowSize = 2 # 2 to the left, 2 to the right
numHiddenUnits = 100
trainFile = 'data/NER-de-train.tsv'
devFile = 'data/NER-de-dev.tsv'
testFile = 'data/NER-de-test.tsv'



print "NER with Keras, only token, window size %d, float: %s" % (windowSize, theano.config.floatX)


#####################
#
# Read in the vocab
#
#####################
print "Read in the vocab"
vocabPath =  'embeddings/GermEval.vocab.gz'

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

            
     
# Read in data   
print "Read in data and create matrices"    
train_sentences = GermEvalReader.readFile(trainFile)
dev_sentences = GermEvalReader.readFile(devFile)
test_sentences = GermEvalReader.readFile(testFile)

# Create numpy arrays
train_x, train_y = GermEvalReader.createNumpyArray(train_sentences, windowSize, word2Idx, label2Idx)
dev_x, dev_y = GermEvalReader.createNumpyArray(dev_sentences, windowSize, word2Idx, label2Idx)
test_x, test_y = GermEvalReader.createNumpyArray(test_sentences, windowSize, word2Idx, label2Idx)


#####################################
#
# Create the Lasagne Network
#
#####################################




# Create the train and predict_labels function
n_in = 2*windowSize+1
n_hidden = numHiddenUnits
n_out = len(label2Idx)

number_of_epochs = 10
minibatch_size = 35
embedding_size = embeddings.shape[1]

x = T.imatrix('x')  # the data, one word+context per row
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        
        
##### -------> Put your model here <----------
        

        
        
number_of_epochs = 10
minibatch_size = 35
print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

#theano.printing.pydotprint(model._predict, outfile="keras.png")  

for epoch in xrange(number_of_epochs):    
    start_time = time.time()
    
    model.fit(train_x, train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=0, shuffle=False)
            

    print "%.2f sec for training" % (time.time() - start_time)   
    
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(model.predict_classes(dev_x, verbose=0), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes(test_x, verbose=0), test_y, idx2Label)

    print "%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test)  
        
