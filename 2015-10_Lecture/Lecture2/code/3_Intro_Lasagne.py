
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

# In[34]:

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

# In[35]:

def build_mlp(n_in, n_hidden, n_out, input_var=None):
    # Put your code here to build the MLP using Lasagne
    


# ## Create the Train Function
# After loading the data and defining the MLP, we can now create the train function.

# In[36]:

# Parameters
n_in = 28*28
n_hidden = 50
n_out = 10

# Create the necessary training and predict labels function




# ## Train the model
# 
# We run the training for some epochs and output the accurarcy of our network

# In[39]:

#Put your code here to train the model using mini batches


# In[ ]:



