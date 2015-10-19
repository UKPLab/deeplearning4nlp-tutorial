# -*- coding: utf-8 -*-
"""
This is the Keras implementation for the GermEval-2014 dataset on NER

This model uses the idea from Collobert et al., Natural Language Processing almost from Scratch.

It implements the window approach with an isolated tag criterion.

For more details on the task see:
https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2014/2014_GermEval_Nested_Named_Entity_Recognition_with_Neural_Networks.pdf

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
# Create the  Network
#
#####################################

n_in = 2*windowSize+1
n_hidden = numHiddenUnits
n_out = len(label2Idx)

number_of_epochs = 10
minibatch_size = 35
       
        
print "Embeddings shape",embeddings.shape

model = Sequential()
# Embeddings layers, lookups the word indices and maps them to their dense vectors. FixedEmbeddings are _not_ updated during training
# If you switch it to an Embedding-Layer, they will be updated (training time increases significant)   
model.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings]))  

# Flatten concatenates the output of the EmbeddingsLayer. EmbeddingsLayer gives us a 5x100 dimension output, Flatten converts it to 500 dim. vector
model.add(Flatten())

# Hidden + Softmax Layer
model.add(Dense(output_dim=n_hidden, activation='tanh'))
model.add(Dense(output_dim=n_out, activation='softmax'))
            
# Use as training function SGD or Adam
model.compile(loss='categorical_crossentropy', optimizer='adam')
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)


print train_x.shape[0], ' train samples'
print train_x.shape[1], ' train dimension'
print test_x.shape[0], ' test samples'

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)

##################################
#
# Training of the Network
#
##################################


        
number_of_epochs = 10
minibatch_size = 35
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
    
        
