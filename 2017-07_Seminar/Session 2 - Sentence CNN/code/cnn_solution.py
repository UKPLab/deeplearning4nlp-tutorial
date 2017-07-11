"""
This implementation is a Convolutional Neural Network for sentence classification.

It uses the same preprocessing of Kim et al., EMNLP 2014, 'Convolutional Neural Networks for Sentence Classification ' (https://github.com/yoonkim/CNN_sentence).

Run the code:
1) Run 'python preprocess.py'. This will preprocess.py the dataset and create the necessary pickle files in the pkl/ folder.
2) Run this code via: python cnn.py


Code was tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 & TensorFlow 1.2.1
- Keras 2.0.5

Data structure:
To run this network / to run a sentence classification using CNNs, the data must be in a certain format. 
The list train_sentences containts the different sentences of your training data. Each word in the training data is converted to
the according word index in the embeddings matrix. An example could look like:
[[1,6,2,1,5,12,42],
 [7,23,56],
 [35,76,23,64,17,97,43,62,47,65]]
 
Here we have three sentences, the first with 7 words, the second with 3 words and the third with 10 words. 
As our network expects a matrix as input for the mini-batchs, we need to bring all sentences to the same length. This is a requirement 
of Theano to run efficiently.  For this we use the function 'sequence.pad_sequences', which adds 0-padding to the matrix. The list/matrix will look after the padding like this:
[[0,0,0,1,6,2,1,5,12,42],
 [0,0,0,0,0,0,0,7,23,56],
 [35,76,23,64,17,97,43,62,47,65]]
 
To make sure that the network does not interpret 0 as some word, we set the embeddings matrix (word_embeddings) such that the 0-column only contains 0. You can check this by outputting word_embeddings[0].


Our labels (y_train) are a 1-dimensional vector containing the binary label for out sentiment classification example.

This code uses the functional API of Keras: https://keras.io/getting-started/functional-api-guide/

It implements roughly the network proposed by Kim et al., Convolutional Neural Networks for Sentence Classification, using convolutions
with several filter lengths. 

Performance after 5 epochs:
Dev-Accuracy: 79.09% (loss: 0.5046)
Test-Accuracy: 77.44% (loss: 0.5163)
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


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
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence



def wordIdxLookup(word, word_idx_map):
    if word in word_idx_map:
        return word_idx_map[word]
    



data = pkl.load(gzip.open("pkl/data.pkl.gz","rb"))
print("data loaded!")


train_labels = data['train']['labels']
train_sentences = data['train']['sentences']

dev_labels = data['dev']['labels']
dev_sentences = data['dev']['sentences']

test_labels = data['test']['labels']
test_sentences = data['test']['sentences']

word_embeddings = data['wordEmbeddings']

# :: Find the longest sentence in our dataset ::
max_sentence_len = 0
for sentence in train_sentences + dev_sentences + test_sentences:
    max_sentence_len = max(len(sentence), max_sentence_len)

print("Longest sentence: %d" % max_sentence_len)
    

    
y_train = np.array(train_labels)
y_dev = np.array(dev_labels)
y_test = np.array(test_labels)

X_train = sequence.pad_sequences(train_sentences, maxlen=max_sentence_len)
X_dev = sequence.pad_sequences(dev_sentences, maxlen=max_sentence_len)
X_test = sequence.pad_sequences(test_sentences, maxlen=max_sentence_len)


print('X_train shape:', X_train.shape)
print('X_dev shape:', X_dev.shape)
print('X_test shape:', X_test.shape)



#  :: Create the network :: 

print('Build model...')

# set parameters:
batch_size = 50

nb_filter = 50
filter_lengths = [1,2,3]
hidden_dims = 100
nb_epoch = 20



words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')

#Our word embedding layer
wordsEmbeddingLayer = Embedding(word_embeddings.shape[0],
                    word_embeddings.shape[1],
                    input_length=max_sentence_len,                   
                    weights=[word_embeddings],
                    trainable=False)

words = wordsEmbeddingLayer(words_input)

#Now we add a variable number of convolutions
words_convolutions = []
for filter_length in filter_lengths:
    words_conv = Convolution1D(filters=nb_filter,
                            kernel_size=filter_length,
                            padding='same',
                            activation='relu',
                            strides=1)(words)
                            
    words_conv = GlobalMaxPooling1D()(words_conv)      
    
    words_convolutions.append(words_conv)  

output = concatenate(words_convolutions)



# We add a vanilla hidden layer:
output = Dropout(0.5)(output)
output = Dense(hidden_dims, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(output)
output = Dropout(0.25)(output)


# We project onto a single unit output layer, and squash it with a sigmoid:
output = Dense(1, activation='sigmoid',  kernel_regularizer=keras.regularizers.l2(0.01))(output)

model = Model(inputs=[words_input], outputs=[output])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

for epoch in range(nb_epoch):
    print("\n------------- Epoch %d ------------" % (epoch+1))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1)
    
    dev_loss, dev_accuracy = model.evaluate(X_dev, y_dev, verbose=False)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
    
  
    print("Dev-Accuracy: %.2f%% (loss: %.4f)" % (dev_accuracy*100, dev_loss))
    print("Test-Accuracy: %.2f%% (loss: %.4f)" % (test_accuracy*100, test_loss))
    
 
    
    
