
# coding: utf-8

# # Introduction to Theano

# For a Theano tutorial please see: http://deeplearning.net/software/theano/tutorial/index.html

# ## Basic Operations

# For more details see: http://deeplearning.net/software/theano/tutorial/adding.html
# 
# **Task**:  Use Theano to compute a simple polynomial function $$f(x,y) = 3x+xy+3y$$
# 
# Hints:
# - First define two input variables with the correct type (http://deeplearning.net/software/theano/library/tensor/basic.html#all-fully-typed-constructors)
# - Define the computation of the function and store it in a variable
# - Use the theano.function() to compile your computation graph
# 

# In[8]:

import theano
import theano.tensor as T

x = T.dscalar('x') #First input variable to the compute graph
y = T.dscalar('y') #Second input variable to the compute graph
z = 3*x + x*y + 3*y #Our formula we like to compute

#Compile for the output z, given the inputs x and y
f = theano.function(inputs=[x,y], outputs=z)


# Now you can invoke f and pass the input values, i.e. f(1,1), f(10,-3) and the result for this operation is returned.

# In[9]:

print f(1,1)
print f(10,-3)


# **Printing of the graph**
# 
# You can print the graph for the above value of z. For details see:
# http://deeplearning.net/software/theano/library/printing.html
# http://deeplearning.net/software/theano/tutorial/printing_drawing.html
# 
# **To print the graph, futher libraries must be installed. In 99% of your development time you don't need the graph printing function. Feel free to skip this section**

# In[10]:

#Graph for z
theano.printing.pydotprint(z, outfile="pics/z_graph.png", var_with_name_simple=True)  

#Graph for function f (after optimization)
theano.printing.pydotprint(f, outfile="pics/f_graph.png", var_with_name_simple=True)  


# **The graph fo z:**
# <img src="files/pics/z_graph.png">
# 
# **The graph for f:**
# <img src="files/pics/f_graph.png">

# ## Simple matrix multiplications

# The following types for input variables are typically used:
# 
#     byte: bscalar, bvector, bmatrix, btensor3, btensor4
#     16-bit integers: wscalar, wvector, wmatrix, wtensor3, wtensor4
#     32-bit integers: iscalar, ivector, imatrix, itensor3, itensor4
#     64-bit integers: lscalar, lvector, lmatrix, ltensor3, ltensor4
#     float: fscalar, fvector, fmatrix, ftensor3, ftensor4
#     double: dscalar, dvector, dmatrix, dtensor3, dtensor4
#     complex: cscalar, cvector, cmatrix, ctensor3, ctensor4
# 
# scalar: One element (one number)
# vector: 1-dimension
# matrix: 2-dimensions
# tensor3: 3-dimensions
# tensor4: 4-dimensions
# 
# As we do not need perfect precision we use mainly float instead of double. Most GPUs are also not able to handle doubles.
# 
# So in practice you need: iscalar, ivector, imatrix and fscalar, fvector, vmatrix.
# 
# **Task**: Implement the function $$f(x,W,b) = \tanh(xW+b)$$ with $x \in \mathbb{R}^n, b \in \mathbb{R}^k, W \in \mathbb{R}^{n \times k}$.
# 
# $n$ input dimension and $k$ output dimension

# In[11]:

import theano
import theano.tensor as T
import numpy as np

x = T.fvector('x')
W = T.fmatrix('W')
b = T.fvector('b')

activation = T.dot(x,W)+b
z = T.tanh(activation)

f = theano.function(inputs=[x,W,b], outputs=[activation,z])


# Next we define some NumPy-Array with data and let Theano compute the result for $f(x,W,b)$

# In[12]:

inputX = np.asarray([0.1, 0.2, 0.3], dtype='float32')
inputW = np.asarray([[0.1,-0.2],[-0.4,0.5],[0.6,-0.7]], dtype='float32')
inputB = np.asarray([0.1,0.2], dtype='float32')

print "inputX.shape",inputX.shape
print "inputW.shape",inputW.shape

f(inputX, inputW, inputB)


# Don't confuse x,W, b with inputX, inputW, inputB. x,W,b contain pointer to your symbols in the compute graph. inputX,inputW,inputB contains your data.

# ## Shared Variables and Updates
# See: http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables
# 
# - Using shared variables, we can create an internal state.
# - Creation of a accumulator:
#     - At the beginning initialize the state to 0
#     - With each function call update the state by certain value
# - Later, in your neural networks, the weight matrices $W$ and the bias values $b$ will be stored as internal state / as shared variable. 
# - Shared variables improve performance, as you need less transfer between your Python code and the execution of the compute graph (which is written & compiled from C code)
# - Shared variables can also be store on your graphic card

# In[13]:

import theano
import theano.tensor as T
import numpy as np

#Define my internal state
init_value = 1

state = theano.shared(value=init_value, name='state')

#Define my operation f(x) = 2*x
x = T.lscalar('x')
z = 2*x

accumulator = theano.function(inputs=[], outputs=z, givens={x: state})

print accumulator()
print accumulator()


# ### Shared Variables
# - We use theano.shared() to share a variable (i.e. make it internally available for Theano)
# - Internal state variables are passed by compile time via the parameter *givens*. So to compute the ouput *z*, use the shared variable *state* for the input variable *x*
# - For information on the borrow=True parameter see: http://deeplearning.net/software/theano/tutorial/aliasing.html
# - In most cases we can set it to true and increase by this the performance.
# 
# 

# ## Updating Shared Variables
# - Using the *updates*-parameter, we can specify how our shared variables should be updated
# - This is useful to create a train function for a neural network. 
#     - We create a function *train(data)* which computes the error and gradient
#     - The computed gradient is then used in the same call to update the shared weights
#     - Training just becomes: *for mini_batch in mini_batches: train(mini_batch)*

# In[14]:

#New accumulator function, now with an update
inc = T.lscalar('inc')
accumulator = theano.function(inputs=[inc], outputs=(state,z), givens={x: state}, updates=[(state,state+inc)])

print accumulator(1)
print accumulator(1)
print accumulator(1)


# - In the above example we increase the state by the variable *inc*
# - The value for *inc* is passed as value to our function

# In[ ]:



