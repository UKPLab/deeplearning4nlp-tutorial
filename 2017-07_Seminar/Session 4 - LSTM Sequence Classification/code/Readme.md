# NER using the Bidirectional LSTM

This is a simple Named Entity Recoginizer for German based on a Bi-Directional LSTM.

We use the data from the GermEval-2014 contest (https://sites.google.com/site/germeval2014ner/data). 

The code was developed and tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 and tensorflow 1.2.1
- Keras 2.0.5

# 1. Step: Word Embeddings
A critical feature for nearly every system in NLP are good word embeddings. For English, there are three pre-trained word embeddings we can use:
- Word2Vec: https://code.google.com/p/word2vec/
- Glove: http://nlp.stanford.edu/projects/glove/
- Levy Word2Vec on Dependencies: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

For German, you can use the word embeddings we trained for the GermEval-2014 contest:
https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/

# 2. Reducing the size of the embedding matrix
The full embedding matrix can become quite large, the unzipped file for the German word embeddings with min. word count of 5 has a size 3.3GB. Reading this and storing it in memory would cost quite some time.

Most of the word embeddings will not be needed during training and evaluation time. So it is a nice trick to first only extract the word embeddings we are going to need for our neural network.  The provided CreateWordList.py (in the Session 1-folder) reads in the dataset and extracts all words from our train, dev and test files.

After that, we can execute the CreateSubCorpus.py (Session 1-folder), which extracts from the large .vocab-file only the word embeddings we actually gonna need.

The reduced embeddings file can be found in at embeddings/GermEval.vocab.gz

# 3. Performance and Runtime
Training LSTMs is quite slow, so bring some patience. 

The on-optimized version achieves the following results (using Theano-Backend):

Single BiLSTM (100 hidden units), after 15 epochs:
Dev-Data: Prec: 0.789, Rec: 0.739, F1: 0.763
Test-Data: Prec: 0.781, Rec: 0.717, F1: 0.747

Stacked BiLSTM (2 layers, 64 hidden states), after 25 epochs:
Dev-Data: Prec: 0.804, Rec: 0.763, F1: 0.783
Test-Data: Prec: 0.795, Rec: 0.738, F1: 0.766

Our system for the GermEval-2014 competition achieved a score of F1=73.5% without any features and F1=77.1% with additional features.
https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2014/2014_GermEval_Nested_Named_Entity_Recognition_with_Neural_Networks.pdf

  
