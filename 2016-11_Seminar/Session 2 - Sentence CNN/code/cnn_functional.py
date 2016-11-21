"""
Code was tested with:
- Python 2.7
- Theano 0.8.2
- Keras 1.1.0

This code uses the functional API of Keras: https://keras.io/getting-started/functional-api-guide/

It implements roughly the network proposed by Kim et al., Convolutional Neural Networks for Sentence Classification, using convolutions
with several filter lengths. 
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K
import cPickle
import gzip
from keras.regularizers import l2, activity_l2



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
                    dropout=0.0,
                    weights=[word_embeddings],
                    trainable=False)

words = wordsEmbeddingLayer(words_input)

#Now we add a variable number of convolutions
words_convolutions = []
for filter_length in filter_lengths:
    words_conv = Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='same',
                            activation='relu',
                            subsample_length=1)(words)
                            
    words_conv = GlobalMaxPooling1D()(words_conv)      
    
    words_convolutions.append(words_conv)  

output = merge(words_convolutions, mode='concat')



# We add a vanilla hidden layer:
output = Dropout(0.5)(output)
output = Dense(hidden_dims, activation='tanh',  W_regularizer=keras.regularizers.l2(0.01))(output)
output = Dropout(0.25)(output)


# We project onto a single unit output layer, and squash it with a sigmoid:
output = Dense(1, activation='sigmoid',  W_regularizer=keras.regularizers.l2(0.01))(output)

model = Model(input=[words_input], output=[output])



model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
    
    
