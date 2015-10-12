
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

# In[1]:

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

            W = theano.shared(value=W_values, name='W')

        if b is None: #Initialize bias to zeor
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        #Compute the activation
        lin_output = T.dot(input, self.W) + self.b 
        
        #Compute the output
        if activation is None:
            self.output = lin_output
        else:
            self.output = activation(lin_output)
        
        
        #Parameters of the model that can be trained
        self.params = [self.W, self.b]


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

# In[2]:

import numpy
import theano
import theano.tensor as T


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), 
                                                dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                dtype=theano.config.floatX), name='b')
            
        #Compute the output of the softmax layer, we call it P(y | x), i.e. how
        #likely is the label y given the input x
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
            
        #For prediction we select the most probable output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            
        # parameters of the model
        self.params = [self.W, self.b]
        
            
    def negative_log_likelihood(self, y):
        """
        Computes the negative log-likelihood. The function explained:
        
        T.log(self.p_y_given_x): Compute the negative log-likelihood of p_y_given_x
        T.arange(y.shape[0]), y]: Select the neuron at position y, our label
        T.mean(): Compute the average over our mini batch
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


# ## MLP
# Our Multi-Layer-Perceptron now plugs everything together, i.e. one hidden layer and the softmax layer.
# 

# In[3]:

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
        self.hiddenLayer = HiddenLayer(rng=rng,
            input=input, n_in=n_in, n_out=n_hidden,
            activation=T.tanh)
        
        self.softmaxLayer = SoftmaxLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden, n_out=n_out)
        
        #Negative log likelihood of this MLP = neg. log likelihood of softmax layer
        self.negative_log_likelihood = self.softmaxLayer.negative_log_likelihood
        
        #Parameters of this MLP = Parameters offen Hidden + SoftmaxLayer
        self.params = self.hiddenLayer.params + self.softmaxLayer.params       
            


# ## Read data + train the network
# Finally we have all blocks to create a MLP for the MNIST dataset.
# 
# You find the MNIST dataset in the data dir. Otherwise you can obtain it from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

# In[ ]:

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
train_x_shared = theano.shared(value=np.asarray(train_x, dtype='float32'), name='train_x')
train_y_shared = theano.shared(value=np.asarray(train_y, dtype='int32'), name='train_y')


print "Shape of train_x-Matrix: ",train_x_shared.get_value().shape
print "Shape of train_y-vector: ",train_y_shared.get_value().shape
print "Shape of dev_x-Matrix: ",dev_x.shape
print "Shape of test_x-Matrix: ",test_x.shape

###########################
#
# Start to build the model
#
###########################

# Hyper parameters
hidden_units = 50
learning_rate = 0.01
batch_size = 20

# Variables for our network
index = T.lscalar()  # index to a minibatch
x = T.fmatrix('x')  # the data, one image per row
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

rng = numpy.random.RandomState(1234) #To have deterministic results

# construct the MLP class
classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=50, n_out=10)

# Define our cost function = error function
cost = classifier.negative_log_likelihood(y) #Here we could add L1 and L2 terms for regularization

# Update param := param - learning_rate * gradient(cost, param)
# See Lecture 1 slide 28
updates = [(param, param - learning_rate * T.grad(cost, param) ) for param in classifier.params]

# Now create a train function
# The train function needs the data, the index for the minibatch and the updates to work correctly
train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x_shared[index * batch_size: (index + 1) * batch_size],
            y: train_y_shared[index * batch_size: (index + 1) * batch_size]
        }
    )

# Create a prediction function
predict_labels = theano.function(inputs=[x], outputs=classifier.softmaxLayer.y_pred)

print ">> train- and predict-functions are compiled <<"


# **Time to train the model**
# 
# Now we can train our model by calling train_model(mini_batch_index). To predict labels, we can use the function predict_labels(data).

# In[ ]:

number_of_minibatches = len(train_x) / batch_size
print "%d mini batches" % (number_of_minibatches)

number_of_epochs = 10
print "%d epochs" % number_of_epochs

#
def compute_accurarcy(dataset_x, dataset_y): 
    predictions = predict_labels(dataset_x)
    errors = sum(predictions != dataset_y) #Number of errors
    accurarcy = 1 - errors/float(len(dataset_y))
    return accurarcy

for epoch in xrange(number_of_epochs):
    #Train the model on all mini batches
    for idx in xrange(0, number_of_minibatches):
        train_model(idx)
 

    accurarcy_dev = compute_accurarcy(dev_x, dev_y)
    accurarcy_test = compute_accurarcy(test_x, test_y)

    print "%d epoch: Accurarcy on dev: %f, accurarcy on test: %f" % (epoch, accurarcy_dev, accurarcy_test)
    
print "DONE"

