# Deep Learning for NLP - Seminar November 2016

This GIT repository accompanies the [UKP](https://www.ukp.tu-darmstadt.de/ukp-home/) seminar on Deep Learning for Natural Language Processing held at the University of Duisburg-Essen.

In contrast to other seminars, this seminar focuses on the **usage of deep learning methods**. As programming infrastructure we use Python in combination with [Theano](http://deeplearning.net/software/theano/) and [Keras](https://keras.io). The published code uses Python 2.7, Theano 0.8.2 and Keras 1.1.1. You should ensure that you have the frameworks installed in the right version (note: they change quickly).

This seminar is structured into 4 sessions:

1. Feed-Forward Networks for Sequence Classification (e.g. POS, NER, Chunking)
2. Convolutional Neural Network for Sentence / Text Classification (e.g. sentiment classification)
3. Convolutional Neural Network for Relation Extraction (e.g. semantic relation extration)
4. Long-Short-Term-Memory (LSTM)-Networks for Sequence Classificaiton 

The seminar is inspired by an engineering mindset: The beautiful math and complexity of the topic is sometimes neglected to provide instead an easy-to-understand and easy-to-use approach **to use Deep Learning for NLP tasks** (we use what works without providing a full background on every aspect).

At the end of the seminar you should be able to understand the most important aspect of deep learning for NLP and be able to programm and train your own deep neural networks.

In case of questions, feel free to approach [Nils Reimers](https://www.ukp.tu-darmstadt.de/people/doctoral-researchers/nils-reimers/).

## Recommended Readings on Deep Learning
The following is a short list with good introductions to different aspects of deep learning.
* 2009, Yoshua Bengio, [Learning Deep Architectures for AI by Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.p)
* 2013, Richard Socher and Christopher Manning, [Deep Learning for Natural Language Processing (slides and recording from NAACL 2013)](http://nlp.stanford.edu/courses/NAACL2013/)
* 2015, Yoshua Bengio et al., [Deep Learning - MIT Press book in preparation](http://www.iro.umontreal.ca/~bengioy/dlbook/)
* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)
 * [2016 videos](https://www.youtube.com/watch?v=kZteabVD8sU&index=1&list=PLCJlDcMjVoEdtem5GaohTC1o9HTTFtK7_)
* 2015, Yoav Goldberg, [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)

## Theory 1 - Introduction to Deep Learning 
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/1_Theory_Introduction.pdf)

The first theory lesson covers the fundamentals of deep learning. 

## Theory 2 - Introduction to Deep Learning Frameworks
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/2_Theory_Frameworks.pdf)

The second lesson gives an overview of deep learning frameworks. Hint: Use [Keras](http://keras.io) and have a look at Theano and TensorFlow.

## Code Session 1 - SENNA Architecture for Sequence Classification
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/Session%201%20-%20SENNA/SENNA.pdf)

**Code:** See folder [Session 1 - SENNA](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar/Session%201%20-%20SENNA)

The first code session is about the SENNA architecture ([Collobert et al., 2011, NLP (almost) from scratch](https://arxiv.org/abs/1103.0398)). In the folder you can find Python code for the preprocessing as well as Keras code to train and evaluate a deep learning model. The folder contains an example for Part-of-Speech tagging, which require the English word embeddings from [Levy et al.](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/). 

You can find in this folder also an example for German NER, based on the [GermEval 2014 dataset](https://sites.google.com/site/germeval2014ner/). To run the German NER code, you need the [word embeddings for German from our website](https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/).

**Recommended Readings:**
 * [CS224d - Lecture 2](https://www.youtube.com/watch?v=T8tQZChniMk)
 * [CS224d - Lecture 3](https://www.youtube.com/watch?v=T1j2Q9_FgTM)

## Theory 3 - Introduction to Convolutional Neural Networks
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/3_Theory_Convolutional_NN.pdf)

This is an introduction to Convolutional Neural Networks.

**Recommended Readings:**
 * [CS224d - Lecture 13](https://www.youtube.com/watch?v=EevTPpQvxiU)
 * [Kim, 2014, Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)


## Code Session 2 - Convolutional Neural Networks for Text Classification
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/Session%202%20-%20Sentence%20CNN/Sentence_CNN.pdf)

**Code:** See folder [Session 2 - Sentence CNN](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar/Session%202%20-%20Sentence%20CNN)

This is a Keras implementation of the [Kim, 2014, Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882). We use the same preprocessing as provided by Kim in his [github repository](https://github.com/yoonkim/CNN_sentence) but then implement the rest using Keras.


## Code Session 3 - Convolutional Neural Networks for Relation Extraction
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/Session%203%20-%20Relation%20CNN/Relation_CNN.pdf)

**Code:** See folder [Session 3 - Relation CNN](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar/Session%203%20-%20Relation%20CNN)

This is an implementation for relation extraction. We use the [SemEval 2010 - Task 8](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview) dataset on semantic relations. We model the task as a pairwise classification task.

**Recommended Readings:**
 * [Zeng et al., 2014, Relation Classification via Convolutional Deep Neural Network](http://www.aclweb.org/anthology/C14-1220)
 * [dos Santos et al., 2015, Classifying Relations by Ranking with Convolutional Neural Networks](https://arxiv.org/abs/1504.06580)


## Theory 4 - Introduction to LSTM
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/4_Theory_Recurrent_Neural_Networks.pdf)

**Code:** See folder [Session 4 - LSTM Sequence Classification](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar/Session%204%20-%20LSTM%20Sequence%20Classification)

LSTMs are a powerful model and became very popular in 2015 / 2016. 

**Recommended Readings:**
  * [RNN Effectivness](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    * [RNN Effectivness - Video](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks)
  * [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  * [C224d Lecture 7](https://www.youtube.com/watch?v=rFVYTydGLr4)

## Code Session 4 - LSTM for Sequence Classification
**Slides:** [pdf](https://github.com/UKPLab/deeplearning4nlp-tutorial/raw/master/2016-11_Seminar/Session%204%20-%20LSTM%20Sequence%20Classification/LSTM%20for%20Sequence%20Classification.pdf)

The folder contains a Keras implementation to perfrom sequence classification using LSTM. We use the [GermEval 2014 dataset](https://sites.google.com/site/germeval2014ner/) for German NER. But you can adapt the code easily to any other sequence classification problem (POS, NER, Chunking etc.). Check the slides for more information.
