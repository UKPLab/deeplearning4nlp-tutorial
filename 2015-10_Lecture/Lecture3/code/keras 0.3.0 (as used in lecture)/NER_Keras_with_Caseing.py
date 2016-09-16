# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


import time
import gzip

import GermEvalReader
import GermEvalReader_with_casing
import BIOF1Validation

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Merge
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding

from KerasLayer.FixedEmbedding import FixedEmbedding


windowSize = 2 # 2 to the left, 2 to the right
numHiddenUnits = 100
trainFile = 'data/NER-de-train.tsv'
devFile = 'data/NER-de-dev.tsv'
testFile = 'data/NER-de-test.tsv'



print "NER with Keras with %s" % theano.config.floatX


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
        
embeddings = np.asarray(embeddings, dtype=theano.config.floatX)

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
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'PADDING':5}
            
caseMatrix = np.identity(len(caseLookup), dtype=theano.config.floatX)
     
# Read in data   
print "Read in data and create matrices"    
train_sentences = GermEvalReader.readFile(trainFile)
dev_sentences = GermEvalReader.readFile(devFile)
test_sentences = GermEvalReader.readFile(testFile)

# Create numpy arrays
train_x, train_case_x, train_y = GermEvalReader_with_casing.createNumpyArrayWithCasing(train_sentences, windowSize, word2Idx, label2Idx, caseLookup)
dev_x, dev_case_x, dev_y = GermEvalReader_with_casing.createNumpyArrayWithCasing(dev_sentences, windowSize, word2Idx, label2Idx, caseLookup)
test_x, test_case_x, test_y = GermEvalReader_with_casing.createNumpyArrayWithCasing(test_sentences, windowSize, word2Idx, label2Idx, caseLookup)



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

dim_case = 6

x = T.imatrix('x')  # the data, one word+context per row
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        
        
print "Embeddings shape",embeddings.shape

words = Sequential()
words.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings]))        #input_length=n_in,
words.add(Flatten())

casing = Sequential()
#casing.add(Embedding(output_dim=dim_case, input_dim=len(caseLookup), input_length=n_in))       
casing.add(FixedEmbedding(output_dim=caseMatrix.shape[1], input_dim=caseMatrix.shape[0], input_length=n_in, weights=[caseMatrix]))       
casing.add(Flatten())

model = Sequential()
model.add(Merge([words, casing], mode='concat'))

model.add(Dense(output_dim=n_hidden, input_dim=n_in*embedding_size, init='uniform', activation='tanh'))
model.add(Dense(output_dim=n_out, init='uniform', activation='softmax'))
            
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


print(train_x.shape[0], 'train samples')
print(train_x.shape[1], 'train dimension')
print(test_x.shape[0], 'test samples')

train_y_cat = np_utils.to_categorical(train_y, n_out)


 
#Function that helps to iterate over our data in minibatches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
        
number_of_epochs = 10
minibatch_size = 35
print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

for epoch in xrange(number_of_epochs):    
    start_time = time.time()
    
    model.fit([train_x, train_case_x], train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=0, shuffle=False)
    #for batch in iterate_minibatches(train_x, train_y_cat, minibatch_size, shuffle=False):
    #    inputs, targets = batch
    #    model.train_on_batch(inputs, targets)   
        

    print "%.2f sec for training" % (time.time() - start_time)
    
  
    
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(model.predict_classes([dev_x, dev_case_x], verbose=0), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes([test_x, test_case_x], verbose=0), test_y, idx2Label)

    print "%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test)
    
        
