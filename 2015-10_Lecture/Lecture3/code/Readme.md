# NER using the SENNA Architecture

This is a simpe Named Entity Recoginizer based on the SENNA architecture as presented by Collobert and Weston in the paper 'Natural Language Processing (almost) from Scratch.

We use the data from the GermEval-2014 contest (https://sites.google.com/site/germeval2014ner/data). 

# 1. Step: Word Embeddings
A critical feature for nearly every system in NLP are good word embeddings. For English, there are three pre-trained word embeddings we can use:
- Word2Vec: https://code.google.com/p/word2vec/
- Glove: http://nlp.stanford.edu/projects/glove/
- Levy Word2Vec on Dependencies: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

For German, you can use the word embeddings we trained for the GermEval-2014 contest:
https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/

# 2. Reducing the size of the embedding matrix
The full embedding matrix can become quite large, the unzipped file for the German word embeddings with min. word count of 5 has a size 3.3GB. Reading this and storing it in memory would cost quite some time.

Most of the word embeddings will not be needed during training and evaluation time. So it is a nice trick to first only extract the word embeddings we are going to need for our neural network.  The provided CreateWordList.py reads in the dataset and extracts all words from our train, dev and test files.

After that, we can execute the CreateSubCorpus.py, which extracts from the large .vocab-file only the word embeddings we actually gonna need.

The reduced embeddings file can be found in at embeddings/GermEval.vocab.gz

# 3. Create a NER
You can use NER_Keras_Skeleton.py or NER_Lasagne_Skeleton.py if you like to start from scratch.

Most of the code deals with reading in the dataset, creating X- and Y-matrices for our neural network and evaluating the final result.

BIOF1Validation.py: Provides methods to compute the F1-score on BIO encoded data
GermEvalReader.py: Reads in the tsv-data from the GermEval task and outputs them as matrices

# 4. Hints
Updating the word embeddings layer takes significant time and is not necessary, as we have pre-trained word embeddings. For lasagne, you should ensure that the word embeddings are not updated, i.e. the W-matrix of the EmbeddingLayer should not be updated by Lasagne.

For Keras we provided a FixedEmbedding implementation (KerasLayer/FixedEmbedding.py). The embedding matrix of this layer will *not* be updated during training time.

# 5. Performance and Runtime
On my computer, the Keras implementation with a window size of 2 runs for about 9.5 seconds per epoch, the Lasagne implementation for about 29.5 seconds per epoch.

Adding the casing information to Keras increases the runtime to about 12-14 seconds/epoch.

The performance after 10 epochs is:
Without case information:
10 epoch: F1 on dev: 0.699177, F1 on test: 0.693651

With case information:
10 epoch: F1 on dev: 0.710590, F1 on test: 0.709284


Our system for the GermEval-2014 competition achieved a score of F1=75.1%
https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2014/2014_GermEval_Nested_Named_Entity_Recognition_with_Neural_Networks.pdf

  
