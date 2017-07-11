# -*- coding: utf-8 -*-
"""
This is an example for performing sequence tagging with Keras.

We use the GermEval 2014 NER dataset (German) and implement the SENNA architecture (Collobert et al., NLP (almost) from scratch).

The code can easily be changed to any other sequence tagging task.

Performance after 10 epochs (GermEval 2014 German NER):
Development F1-score: 70.3%
Test F1-score: 69.9%



Code was written & tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 and tensorflow 1.2.1
- Keras 2.0.5

@author: Nils Reimers, www.deeplearning4nlp.com
"""
from __future__ import print_function
import numpy as np
import time
import gzip

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl


import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding

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



print(train_tokens.shape[0], ' train samples')
print(train_tokens.shape[1], ' train dimension')
print(test_tokens.shape[0], ' test samples')



# !!TODO!!
# Add your Keras network here
# !! / TODO !!




##################################
#
# Training of the Network
#
##################################


        
number_of_epochs = 10
minibatch_size = 128
print("%d epochs" % number_of_epochs)


def predict_classes(prediction):
 return prediction.argmax(axis=-1)
 
for epoch in range(number_of_epochs):
    print("\n------------- Epoch %d ------------" % (epoch+1))
    model.fit([train_tokens, train_case], train_y, epochs=1, batch_size=minibatch_size, verbose=True, shuffle=True)   
    
    
    # Compute precision, recall, F1 on dev & test data
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(predict_classes(model.predict([dev_tokens, dev_case])), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(predict_classes(model.predict([test_tokens, test_case])), test_y, idx2Label)

    print("%d. epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test))
    
  
