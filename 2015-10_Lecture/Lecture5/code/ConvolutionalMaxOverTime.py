from __future__ import absolute_import
import theano
import theano.tensor as T
import keras
from keras import activations, initializations, regularizers, constraints
from keras.layers.core import Layer, MaskedLayer
from keras.utils.theano_utils import sharedX
from keras.constraints import unitnorm
from keras.utils.theano_utils import shared_zeros, floatX, ndim_tensor


"""
This is a quick and dirty implementation of a convolutional layer + max over time (as described by
Collobert et al., NLP almost from scratch, section 3.2.2 sentence approach).
This layer is NOT tested at all and contains probably bugs. I took the implementation from 
keras.core.Dense and adapted it to the new use case.

This layer expects a 3 dimensional input in the format (nb_samples, nb_time_steps, input_dim)
and outputs a 2 dimensional vector in the format (nb_samples, output_dim)

It would be nice if this layer could be extended to a larger window size, in the provided
IMDB_ConvMaxOverTime script it only looks at unigram words. 

@author: Nils Reimers
"""
class ConvolutionalMaxOverTime(Layer):
    input_ndim = 3

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ConvolutionalMaxOverTime, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        self.input = T.matrix()
        self.W = self.init((input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim,))

        self.params = [self.W, self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = T.dot(X, self.W) 
        
        max_over_time = T.max(output,axis=1)  + self.b
        
        return self.activation(max_over_time)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim}
        base_config = super(ConvolutionalMaxOverTime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

    
