
# coding: utf-8

# # Handwritten Digit Recognition with Theano 
# 
# In this tutorial we will train a feed forward network / multi-layer-perceptron (MLP) to recognize handwritten digits using pure Theano.
# For a long version see: http://deeplearning.net/tutorial/mlp.html
# 
# 
# 

# ## Layout
# The layout of our network 
# <img src="http://deeplearning.net/tutorial/_images/mlp.png">
# Source of image: http://deeplearning.net/tutorial/mlp.html
# 
# Our networks has 3 layers
# - Input layer, $28*28=786$ dimensional (the pixels of the images)
# - A hidden layer
# - A Softmax layer
# 
# In order to make our lives easier, we will create the following files / classes / components:
# - HiddenLayer - To model a hidden layer
# - SoftmaxLayer - To model a softmax layer
# - MLP - Combines several hidden & softmax layers together to form a MLP
# - One file for reading the data and training the network
# 

# ## HiddenLayer
# 
# The hidden layer computes the following function:
# $$\text{output} = \tanh(xW + b)$$
# 
# The matrix $W$ will be initialized Glorot-style (see 1. Lecture).
# 
# This is the class we will use for the hidden layer:

# In[58]:

import numpy 
import theano
import theano.tensor as T
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        :param rng: Random number generator, for reproducable results
        :param input: Symbolic Theano variable for the input
        :param n_in: Number of incoming units
        :param n_out: Number of outgoing units
        :param W: Weight matrix
        :param b: Bias
        :param activation: Activation function to use
        """
        self.input = input
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.activation=activation
        
        
        if W is None: #Initialize Glorot Style
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid or activation == theano.tensor.nnet.hard_sigmoid or activation == theano.tensor.nnet.ultra_fast_sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None: #Initialize bias to zeor
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # Put your code here: Implement a function to compute activation(x*W+b)
        


# ## Softmax Layer
# The softmax-layer computes: 
# $$\text{output} = \text{softmax}(xW+b)$$
# 
# As for the hidden layer, we allow the parameterization of the number of neurons. The weight matrix and bias vector is initialized to zero.
# 
# As we performt a single label classification task, we use the negative log-likelihood as error function:
# $$E(x,W,b) = -log(o_y)$$
# 
# with $o_y$ the output for label $y$.

# In[59]:

import numpy
import theano
import theano.tensor as T


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), 
                                                dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                dtype=theano.config.floatX), name='b', borrow=True)
            
        # Put your code here, implement a function to compute softmax(x*W+b)


# ## MLP
# Our Multi-Layer-Perceptron now plugs everything together, i.e. one hidden layer and the softmax layer.
# 

# In[60]:

import numpy
import theano
import theano.tensor as T

class MLP(object):
     def __init__(self, rng, input, n_in, n_hidden, n_out):
        """
        :param rng: Our random number generator
        :param input: Input variable (the data)
        :param n_in: Input dimension
        :param n_hidden: Hidden size
        :param n_out: Output size
        """
        #Put your code here to build the neural network       
            


# ## Read data + train the network
# Finally we have all blocks to create a MLP for the MNIST dataset.
# 
# You find the MNIST dataset in the data dir. Otherwise you can obtain it from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

# In[61]:

import cPickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T


# Load the pickle file for the MNIST dataset.
dataset = 'data/mnist.pkl.gz'

f = gzip.open(dataset, 'rb')
train_set, dev_set, test_set = cPickle.load(f)
f.close()

#train_set contains 2 entries, first the X values, second the Y values
train_x, train_y = train_set
dev_x, dev_y = dev_set
test_x, test_y = test_set

#Created shared variables for these sets (for performance reasons)
train_x_shared = theano.shared(value=np.asarray(train_x, dtype='float32'), name='train_x', borrow=True)
train_y_shared = theano.shared(value=np.asarray(train_y, dtype='int32'), name='train_y', borrow=True)


print "Shape of train_x-Matrix: ",train_x_shared.get_value().shape
print "Shape of train_y-vector: ",train_y_shared.get_value().shape
print "Shape of dev_x-Matrix: ",dev_x_shared.get_value().shape
print "Shape of test_x-Matrix: ",test_x_shared.get_value().shape

###########################
#
# Start to build the model
#
###########################

# Hyper parameters
hidden_units = 50
learning_rate = 0.01
batch_size = 20

# Put your code here to build the training and predict_labels function


# **Time to train the model**
# 
# Now we can train our model by calling train_model(mini_batch_index). To predict labels, we can use the function predict_labels(data).

# In[62]:

# Train your network on mini batches

