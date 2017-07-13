# Deep Learning for NLP - July 2017

This GIT repository accompanies the seminar on Deep Learning for Natural Language Processing.

In contrast to other seminars, this seminar focuses on the **usage of deep learning methods**. As programming infrastructure we use Python in combination with [Keras](https://keras.io). The published code can be used with Python 2.7 or Python 3.6, Keras 2.0.5, and Theano (0.9.0) or TensorFlow (1.2.1) backend. You should ensure that you have the frameworks installed in the right version (note: they change quickly).

This seminar is structured into 4 sessions:

1. Feed-Forward Networks for Sequence Classification (e.g. POS, NER, Chunking)
2. Convolutional Neural Network for Sentence / Text Classification (e.g. sentiment classification)
3. Convolutional Neural Network for Relation Extraction (e.g. semantic relation extration)
4. Long-Short-Term-Memory (LSTM)-Networks for Sequence Classificaiton 

The seminar is inspired by an engineering mindset: The beautiful math and complexity of the topic is sometimes neglected to provide instead an easy-to-understand and easy-to-use approach **to use Deep Learning for NLP tasks** (we use what works without providing a full background on every aspect).

At the end of the seminar you should be able to understand the most important aspect of deep learning for NLP and be able to programm and train your own deep neural networks.

In case of questions, feel free to approach [Nils Reimers](https://www.ukp.tu-darmstadt.de/people/doctoral-researchers/nils-reimers/).

# Setting up the Development Environment

The codes in this folder were developed for Python 2.7 (and Python 3.6), Keras 2.0.5. As backend for Keras you can use Theano 0.9.0 or TensorFlow 1.2.1.

You can setup a virtual environment in the following way:

You can install the required packages in the following way:
```
pip install -r requirements.txt
```

Alternatively, it should be sufficient to just install Keras in version 2.0.5 and TensorFlow:
```
pip install Keras==2.0.5 TensorFlow==1.2.1
```

## Virtual Environment
It can be useful, to run Python in a virtual environment for this seminar. 

Create a virtualenv in the following way:
```
virtualenv .env
source .env/bin/activate
```
If you operate in the virtual environment, you can run pip to install the needed packages in the following way:
```
.env/bin/pip install -r requirements.txt
```


## Docker
The folder `docker` contains a dockerfile that bundles an environment needed to run the experiments in this folder. It installs Python 3.6, Keras 2.0.5 and Tensorflow 1.2.1.

First, you need to build the docker container:
```
docker build ./docker -t dl4nlp
```

Than, in this folder you can start the container and mount files in this container to the docker container:
```
docker run -it -v ${PWD}:/usr/src/app dl4nlp bash
```

This will start a bash inside the dl4nlp container, where Python is installed. Through the mounting, you can modify the files in this folder and run them inside the docker container.


## Recommended Readings on Deep Learning
The following is a short list with good introductions to different aspects of deep learning.
* 2009, Yoshua Bengio, [Learning Deep Architectures for AI by Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.p)
* 2013, Richard Socher and Christopher Manning, [Deep Learning for Natural Language Processing (slides and recording from NAACL 2013)](http://nlp.stanford.edu/courses/NAACL2013/)
* 2015, Yoshua Bengio et al., [Deep Learning - MIT Press book in preparation](http://www.iro.umontreal.ca/~bengioy/dlbook/)
* 2015, Richard Socher, [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)
* 2015, Yoav Goldberg, [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)

## Theory 1 - Introduction to Deep Learning 
**Slides:** [pdf](./1_Theory_Introduction.pdf)

The first theory lesson covers the fundamentals of deep learning. 

## Theory 2 - Introduction to Word Embeddings
**Slides:** [pdf](./2_Theory_Word_Embeddings.pdf)

## Theory 3 - Introduction to Deep Learning Frameworks
**Slides:** [pdf](./3_Theory_Frameworks.pdf)

The second lesson gives an overview of deep learning frameworks. Hint: Use [Keras](http://keras.io) and have a look at Theano and TensorFlow.

## Code Session 1 - SENNA Architecture for Sequence Classification
**Slides:** [pdf](./Session%201%20-%20SENNA/SENNA.pdf)

**Code:** See folder [Session 1 - SENNA](./Session%201%20-%20SENNA)

The first code session is about the SENNA architecture ([Collobert et al., 2011, NLP (almost) from scratch](https://arxiv.org/abs/1103.0398)). In the folder you can find Python code for the preprocessing as well as Keras code to train and evaluate a deep learning model. The folder contains an example for Part-of-Speech tagging, which require the English word embeddings from either [Levy et al.](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) or from [Komninos et al.](https://www.cs.york.ac.uk/nlp/extvec/). 

You can find in this folder also an example for German NER, based on the [GermEval 2014 dataset](https://sites.google.com/site/germeval2014ner/). To run the German NER code, you need the [word embeddings for German from our website](https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/).

**Recommended Readings:**
 * [CS224d - Lecture 2](https://www.youtube.com/watch?v=T8tQZChniMk)
 * [CS224d - Lecture 3](https://www.youtube.com/watch?v=T1j2Q9_FgTM)

## Theory 4 - Introduction to Convolutional Neural Networks
**Slides:** [pdf](./4_Theory_Convolutional_NN.pdf)

This is an introduction to Convolutional Neural Networks.

**Recommended Readings:**
 * [CS224d - Lecture 13](https://www.youtube.com/watch?v=EevTPpQvxiU)
 * [Kim, 2014, Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)


## Code Session 2 - Convolutional Neural Networks for Text Classification
**Slides:** [pdf](./Session%202%20-%20Sentence%20CNN/Sentence_CNN.pdf)

**Code:** See folder [Session 2 - Sentence CNN](./Session%202%20-%20Sentence%20CNN)

This is a Keras implementation of the [Kim, 2014, Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882). We use the same preprocessing as provided by Kim in his [github repository](https://github.com/yoonkim/CNN_sentence) but then implement the rest using Keras.


## Code Session 3 - Convolutional Neural Networks for Relation Extraction
**Slides:** [pdf](./Session%203%20-%20Relation%20CNN/Relation_CNN.pdf)

**Code:** See folder [Session 3 - Relation CNN](./Session%203%20-%20Relation%20CNN)

This is an implementation for relation extraction. We use the [SemEval 2010 - Task 8](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview) dataset on semantic relations. We model the task as a pairwise classification task.

**Recommended Readings:**
 * [Zeng et al., 2014, Relation Classification via Convolutional Deep Neural Network](http://www.aclweb.org/anthology/C14-1220)
 * [dos Santos et al., 2015, Classifying Relations by Ranking with Convolutional Neural Networks](https://arxiv.org/abs/1504.06580)


## Theory 5 - Introduction to LSTM
**Slides:** [pdf](./5_Theory_Recurrent_Neural_Networks.pdf)

**Code:** See folder [Session 4 - LSTM Sequence Classification](./Session%204%20-%20LSTM%20Sequence%20Classification)

LSTMs are a powerful model and became very popular in 2015 / 2016. 

**Recommended Readings:**
  * [RNN Effectivness](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    * [RNN Effectivness - Video](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks)
  * [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  * [C224d Lecture 7](https://www.youtube.com/watch?v=rFVYTydGLr4)

## Code Session 4 - LSTM for Sequence Classification
**Slides:** [pdf](./Session%204%20-%20LSTM%20Sequence%20Classification/LSTM%20for%20Sequence%20Classification.pdf)

The folder contains a Keras implementation to perfrom sequence classification using LSTM. We use the [GermEval 2014 dataset](https://sites.google.com/site/germeval2014ner/) for German NER. But you can adapt the code easily to any other sequence classification problem (POS, NER, Chunking etc.). Check the slides for more information.




