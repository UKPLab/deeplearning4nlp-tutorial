# -*- coding: utf-8 -*-
"""
This is the Keras implementation for the GermEval-2014 dataset on NER

This model uses the idea from Collobert et al., Natural Language Processing almost from Scratch.

It implements the window approach with an isolated tag criterion.

For more details on the task see:
https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2014/2014_GermEval_Nested_Named_Entity_Recognition_with_Neural_Networks.pdf


!! This is a blank skeleton for your neural network. You can start your assignment here to implement a Neural Network
that performs NER !!



Code was written & tested with:
- Python 2.7
- Theano 0.8.x
- Keras 1.0.x

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


#Casing matrix
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING':7}
                      
caseMatrix = np.identity(len(caseLookup), dtype=theano.config.floatX)
            
     
# Read in data   
print "Read in data and create matrices"    
train_sentences = GermEvalReader.readFile(trainFile)
dev_sentences = GermEvalReader.readFile(devFile)
test_sentences = GermEvalReader.readFile(testFile)

# Create numpy arrays
train_x, train_case_x, train_y = GermEvalReader.createNumpyArrayWithCasing(train_sentences, windowSize, word2Idx, label2Idx, caseLookup)
dev_x, dev_case_x, dev_y = GermEvalReader.createNumpyArrayWithCasing(dev_sentences, windowSize, word2Idx, label2Idx, caseLookup)
test_x, test_case_x, test_y = GermEvalReader.createNumpyArrayWithCasing(test_sentences, windowSize, word2Idx, label2Idx, caseLookup)



#####################################
#
# Create the  Network
#
#####################################


#
# :: Create your Keras network here ::
#


##################################
#
# Training of the Network
#
##################################


        
number_of_epochs = 10
minibatch_size = 64
print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)



for epoch in xrange(number_of_epochs):    
    start_time = time.time()
    
    #Train for 1 epoch
    model.fit(train_x, train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=False, shuffle=True)   
    print "%.2f sec for training" % (time.time() - start_time)
    
  
    # Compute precision, recall, F1 on dev & test data
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(model.predict_classes(dev_x, verbose=0), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes(test_x, verbose=0), test_y, idx2Label)

    print "%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test)
    
        
