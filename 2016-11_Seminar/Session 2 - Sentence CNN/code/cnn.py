"""
This implementation is a Convolutional Neural Network for sentence classification.

It uses the same preprocessing of Kim et al., EMNLP 2014, 'Convolutional Neural Networks for Sentence Classification ' (https://github.com/yoonkim/CNN_sentence).


Preprocessing Option 1:
- Unzip the kim_et_al_preprocessed.p.gz 

Preprocessing Option 2:
1. Download Kim et al. source code at Convolutional Neural Networks for Sentence Classification 
2. Download the word2vec embeddings GoogleNews-vectors-negative300.bin from https://code.google.com/archive/p/word2vec/
3. Run the preprocessing of Kim et al: 'python process_data.py path' where where path points to the word2vec binary file (i.e. GoogleNews-vectors-negative300.bin file). This will create a pickle object called mr.p in the same folder, which contains the dataset in the right format.
4. Copy the pickle object 'mr.p' and store it in this folder. Rename it to kim_et_al_preprocessed.p

Run this code:
Run this code via 'python cnn.py'.

Code was tested with:
- Python 2.7
- Theano 0.8.2
- Keras 1.1.0

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
"""


import numpy as np
from hgext.graphlog import revset
from twisted.positioning.test import test_sentence
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K
import cPickle
import gzip



def wordIdxLookup(word, word_idx_map):
    if word in word_idx_map:
        return word_idx_map[word]
    

# Load the preprocessed data from Kim et al. scripts
#sentences: Reviews, dictionary with the entries 
# -"y": label, 
# -"text": orig_rev,                             
# -"num_words",
# -"split": np.random.randint(0,cv)} 

#word_embeddings: Word Embeddings
#random_embeddings: Random word Embeddings
#word_idx_map: Mapping of words to indices
#vocab: Vocabulary

sentences, word_embeddings, random_embeddings, word_idx_map, vocab = cPickle.load(gzip.open("kim_et_al_preprocessed.p.gz","rb"))
print "data loaded!"



train_labels = []
train_sentences = []

test_labels = []
test_sentences = []

max_sentence_len = 0

for datum in sentences:
    label = datum['y']
    cv = datum['split']
    words = datum['text'].split()    
    wordIndices = [wordIdxLookup(word, word_idx_map) for word in words]
    
    if cv == 0: #CV=0 is our test set
        test_labels.append(label)
        test_sentences.append(wordIndices)
    else:
        train_labels.append(label)
        train_sentences.append(wordIndices)   
        
    max_sentence_len = max(max_sentence_len, len(words))
    
    

    
y_train = np.array(train_labels)
y_test = np.array(test_labels)

X_train = sequence.pad_sequences(train_sentences, maxlen=max_sentence_len)
X_test = sequence.pad_sequences(test_sentences, maxlen=max_sentence_len)


print 'X_train shape:', X_train.shape 
print 'X_test shape:', X_test.shape 



#  :: Create the network :: 

print 'Build model...'

# set parameters:
batch_size = 32

nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 20



model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into our word embeddings 
model.add(Embedding(word_embeddings.shape[0],
                    word_embeddings.shape[1],
                    input_length=max_sentence_len,
                    dropout=0.2,
                    weights=[word_embeddings],
                    trainable=False)) #Set to true, to update word embeddings while training


# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))

# we use max over time pooling:
model.add(GlobalMaxPooling1D())


# We add a vanilla hidden layer:
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.2))


# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
    
    
