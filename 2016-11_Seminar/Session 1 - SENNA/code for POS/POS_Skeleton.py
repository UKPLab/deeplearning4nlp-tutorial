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
import theano
import theano.tensor as T


import time
import gzip
import cPickle as pkl


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Merge
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding





numHiddenUnits = 20


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


#
# ::::: Create your network here !! :::::::::
#


##################################
#
# Training of the Network
#
##################################


#
# :::: Put your train code here :::::::::
#
    
  
