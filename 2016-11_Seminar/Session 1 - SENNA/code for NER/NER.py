# -*- coding: utf-8 -*-
"""
This is an example code for a POS tagger using the SENNA architecture (Collobert et al.) and Keras.


Baseline:
NLTK Uni-/Bi-/Trigram Tagger: 91.33%

Performance after 4 epochs:
Dev-Accuracy: 96.55%
Test-Accuracy: 96.51%



@author: Nils Reimers

Code was written & tested with:
- Python 2.7
- Theano 0.8.1
- Keras 1.1.1

"""
import numpy as np


import time
import gzip
import cPickle as pkl


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Merge
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding

import BIOF1Validation



numHiddenUnits = 100


f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']
caseEmbeddings = embeddings['caseEmbeddings']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('pkl/data.pkl.gz', 'rb')
train_tokens, train_case, train_y = pkl.load(f)
dev_tokens, dev_case, dev_y = pkl.load(f)
test_tokens, test_case, test_y = pkl.load(f)
f.close()

#####################################
#
# Create the  Network
#
#####################################




# Create the train and predict_labels function
n_in = train_tokens.shape[1]
n_hidden = numHiddenUnits
n_out = len(label2Idx)


words = Sequential()
words.add(Embedding(output_dim=wordEmbeddings.shape[1], input_dim=wordEmbeddings.shape[0], input_length=n_in,  weights=[wordEmbeddings], trainable=False))       
words.add(Flatten())

casing = Sequential()  
casing.add(Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], input_length=n_in, weights=[caseEmbeddings], trainable=False))     
casing.add(Flatten())

model = Sequential()
model.add(Merge([words, casing], mode='concat'))

model.add(Dense(output_dim=n_hidden, activation='tanh'))
model.add(Dense(output_dim=n_out, activation='softmax'))
            
            
# Use Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)


print train_tokens.shape[0], ' train samples'
print train_tokens.shape[1], ' train dimension'
print test_tokens.shape[0], ' test samples'



##################################
#
# Training of the Network
#
##################################


        
number_of_epochs = 10
minibatch_size = 128
print "%d epochs" % number_of_epochs

 
for epoch in xrange(number_of_epochs):
    print "\n------------- Epoch %d ------------" % (epoch+1)
    model.fit([train_tokens, train_case], train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=True, shuffle=True)   
    
    
    # Compute precision, recall, F1 on dev & test data
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(model.predict_classes([dev_tokens, dev_case], verbose=0), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes([test_tokens, test_case], verbose=0), test_y, idx2Label)

    print "%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test)
    
  
