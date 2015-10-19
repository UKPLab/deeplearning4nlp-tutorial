# -*- coding: utf-8 -*-
"""
This is the Lasagne implementation for the GermEval-2014 dataset on NER

I would recommend that you use Keras (http://keras.io) instead of Lasagne due to 
a better performance. For the Keras implementation, see NER_Keras.py

@author: Nils Reimers
"""
import numpy as np
import theano
import theano.tensor as T

import lasagne
import time
import gzip

import GermEvalReader
import BIOF1Validation


windowSize = 2 # 2 to the left, 2 to the right
numHiddenUnits = 100
trainFile = 'data/NER-de-train.tsv'
devFile = 'data/NER-de-dev.tsv'
testFile = 'data/NER-de-test.tsv'

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

def build_network(input_var, n_in, n_hidden, n_out, embedding_matrix):
    embedding_size = embedding_matrix.shape[1]
    #Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, n_in), input_var=input_var)
    
    #Embedding lookup layer
    l_embeddings = lasagne.layers.EmbeddingLayer(incoming=l_in, input_size=embedding_matrix.shape[0], 
                                                 output_size=embedding_size, W=embedding_matrix)
    
    #Hidden layer
    l_hid = lasagne.layers.DenseLayer(incoming=l_embeddings, #lasagne.layers.DenseLayer
                num_units=n_in*embedding_size, nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform())
    
    # Our output layer (a softmax layer)
    l_out = lasagne.layers.DenseLayer(incoming=l_hid, #lasagne.layers.DenseLayer 
            num_units=n_out, nonlinearity=lasagne.nonlinearities.softmax)
    
    params = l_out.get_params() + l_hid.get_params()
    
    return (l_out, params)


# Create the train and predict_labels function
n_in = 2*windowSize+1
n_hidden = numHiddenUnits
n_out = len(label2Idx)

x = T.imatrix('x')  # the data, one word+context per row
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

network, params = build_network(x, n_in, n_hidden, n_out, embeddings)

# Create a loss expression for training (cross entropy)
prediction = lasagne.layers.get_output(network)


# Cross Entropy as loss function
#loss = lasagne.objectives.categorical_crossentropy(prediction, y)
#loss = loss.mean()

#Negative Log-Likelihood
loss = -T.mean(T.log(prediction)[T.arange(y.shape[0]), y])

#Get the updates for training
print params

#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.1)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.1)

# Predict the labels
network_predict_label = T.argmax(lasagne.layers.get_output(network, deterministic=True), axis=1)



print "-- Compile functions--"
# Compile a function performing a training step on a mini-batch
train_fn = theano.function(inputs=[x, y], outputs=loss, updates=updates) #



# Create the predict_labels function
predict_labels = theano.function(inputs=[x], outputs=network_predict_label)

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
    for batch in iterate_minibatches(train_x, train_y, minibatch_size, shuffle=True):
        inputs, targets = batch
        train_fn(inputs, targets)              
    
    print "%.2f sec for training" % (time.time() - start_time)

    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(predict_labels(dev_x), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(predict_labels(test_x), test_y, idx2Label)

    print "%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test)
    


print "--DONE--"
            
    
    
    
        
