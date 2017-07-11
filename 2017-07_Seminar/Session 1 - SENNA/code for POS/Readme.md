# POS using the SENNA Architecture

This is a simple POS tagger for English based on the SENNA architecture as presented by Collobert and Weston in the paper 'Natural Language Processing (almost) from Scratch.

We use the data from the Brown Corpus as found in NLTK (http://www.nltk.org). 

The code was developed and tested with:
- Python 2.7
- Theano 0.8.2
- Keras 1.1.1

# 1. Step: Word Embeddings
A critical feature for nearly every system in NLP are good word embeddings. For English, there are three pre-trained word embeddings we can use:
- Word2Vec: https://code.google.com/p/word2vec/
- Glove: http://nlp.stanford.edu/projects/glove/
- Levy Word2Vec on Dependencies: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

My personal perference are the word embeddings from Levy et al. In different of my experiments, they gave the best performance. However, try to test all those word embeddings on your specific task.

For German, you can use the word embeddings we trained for the GermEval-2014 contest:
https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/

# 2. Preprocessing
After you downloaded the Levy word embeddings (deps.words.bz2), open `preprocess.py` and change the path in 'embeddingsPath' to your local copy. After that, you can run this script by executing `python preprocess.py`


The file will create several numpy matrices and store them using cPickle in the `pkl` folder. 

# 3. Running the code
To train and evaluate the POS-tagger, run `python POS.py`. 

This file reads in the cPickle files from the pkl-Folder, creates the neural network using Keras, and train and evaluates it on the provided dataset.


# 4. Performance
The performance after 4 epochs is:
Dev-Accuracy: 96.55
Test-Accuracy: 96.51

As comparison, using a frequence distribute on uni/bi-/trigrams as found in the NLTK book, the accuracy is around 91.33%.

Note: Part-of-Speech tagging is a rather simple task and the performance gain from deep learning is minor.

  
