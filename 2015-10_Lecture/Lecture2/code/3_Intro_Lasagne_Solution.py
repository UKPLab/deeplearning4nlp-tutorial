
# coding: utf-8

# # Introduction to Lasagne
# 
# There are various libaries building on top of Theano to provide easy buidling blocks for designing deep neural networks. Some of them are:
# - Lasagne (https://github.com/Lasagne/Lasagne)
# - Blocks (https://github.com/mila-udem/blocks)
# - Keras (http://keras.io/)
# - OpenDeep (http://www.opendeep.org/)
# 
# All libaries are kind of similar but different in the details, for example in the design philosophy. I chose (after too little research) Lasagne as it will allow you to interact with Theano and the computation graph. Keep an eye onto this evolving area.
# 
# For a great example how to use Lasagne for MNIST see the Lasagne Tutorial: http://lasagne.readthedocs.org/en/latest/user/tutorial.html

# ## Bascis
# Lasagne provides you with several basic components to build your neural networks. Instead of defining your HiddenLayer and SoftmaxLayer as in the previous example, you can use existent implementations from the library and easily plug them together.
# 
# In the following we will reimplement the MLP for the MNIST-dataset using Lasagne. For more information on Lasagne see http://lasagne.readthedocs.org/en/latest/

# ## Load your dataset
# As before we load our dataset. See 2_MNIST for more details.

# In[1]:

import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T

import lasagne

# Load the pickle file for the MNIST dataset.
dataset = 'data/mnist.pkl.gz'

f = gzip.open(dataset, 'rb')
train_set, dev_set, test_set = cPickle.load(f)
f.close()

#train_set contains 2 entries, first the X values, second the Y values
train_x, train_y = train_set
dev_x, dev_y = dev_set
test_x, test_y = test_set


# ## Build the MLP
# Now we use the provided layers from Lasagne to build our MLP

# In[2]:

def build_mlp(n_in, n_hidden, n_out, input_var=None):
    #Input layer, 1 dimension = number of samples, 2 dimension = input, our 28*28 image
    l_in = lasagne.layers.InputLayer(shape=(None, n_in), input_var=input_var)
    
    # Our first hidden layer with n_hidden units
    # As nonlinearity we use tanh, you could also try rectify
    l_hid1 = lasagne.layers.DenseLayer(incoming=l_in,
                num_units=n_hidden, nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform())
    
    # Our output layer (a softmax layer)
    l_out = lasagne.layers.DenseLayer(incoming=l_hid1, 
            num_units=n_out, nonlinearity=lasagne.nonlinearities.softmax)
    
    return l_out
    


# ## Create the Train Function
# After loading the data and defining the MLP, we can now create the train function.

# In[3]:

# Parameters
n_in = 28*28
n_hidden = 50
n_out = 10

# Create the network
x = T.dmatrix('x')  # the data, one image per row
y = T.lvector('y')  # the labels are presented as 1D vector of [int] labels

network = build_mlp(n_in, n_hidden, n_out, x)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, y)
loss = loss.mean()

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)


# Predict the labels
network_predict_label = T.argmax(lasagne.layers.get_output(network, deterministic=True), axis=1)


# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function(inputs=[x, y], outputs=loss, updates=updates)

# Create the predict_labels function
predict_labels = theano.function(inputs=[x], outputs=network_predict_label)




# ## Train the model
# 
# We run the training for some epochs and output the accurarcy of our network

# In[4]:

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

#Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_accurarcy(dataset_x, dataset_y): 
    predictions = predict_labels(dataset_x)
    errors = sum(predictions != dataset_y) #Number of errors
    accurarcy = 1 - errors/float(len(dataset_y))
    return accurarcy

number_of_epochs = 10
print "%d epochs" % number_of_epochs

for epoch in xrange(number_of_epochs):    
    for batch in iterate_minibatches(train_x, train_y, 20, shuffle=True):
        inputs, targets = batch
        train_fn(inputs, targets)      

    accurarcy_dev = compute_accurarcy(dev_x, dev_y)
    accurarcy_test = compute_accurarcy(test_x, test_y)

    print "%d epoch: Accurarcy on dev: %f, accurarcy on test: %f" % (epoch, accurarcy_dev, accurarcy_test)
    
print "DONE"


# In[ ]:



