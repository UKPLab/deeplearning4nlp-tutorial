# Deep Learning for NLP - Lecture October 2015
**This site can be access by the URL: www.deeplearning4nlp.com**

**>> Link for 1. lecture: https://youtu.be/AmG4jzmBZ88 <<**


This GIT repository accompanies the [UKP](https://www.ukp.tu-darmstadt.de/ukp-home/) lecture on Deep Learning for Natural Language Processing.

In contrast to other lectures, this lecture focuses on the **usage of deep learning methods**. As programming infrastructure we use Python in combination with [Theano](http://deeplearning.net/software/theano/) and [Lasagne](https://github.com/Lasagne/Lasagne).

This lecture is structured into 6 parts. Each parts contains some recommended readings, which are supposed to be read before class. In class (video will be streamed and recorded) we will discuss the papers and provide some more background knowledge. With the start of the second lecture, each lecture will contain some practical exercise, in the most cases to implement a certain deep neural network to do a typical NLP task, for example Named Entity Recognition, Genre Classifcation of Sentiment Classification. The lecture is inspired by an engineering mindset: The beautiful math and complexity of the topic is sometimes neglected to provide instead an easy-to-understand and easy-to-use approach **to use Deep Learning for NLP tasks** (we use what works without providing a full background on every aspect).

At the end of the lecture you should be able to understand the most important aspect of deep learning for NLP and be able to programm and train your own deep neural networks.

In case of questions, feel free to approach [Nils Reimers](https://www.ukp.tu-darmstadt.de/people/doctoral-researchers/nils-reimers/).

## Recommended Readings on Deep Learning
The following is a short list with good introductions to different aspects of deep learning.
* 2009, Yoshua Bengio, [Learning Deep Architectures for AI by Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.p)
* 2013, Richard Socher and Christopher Manning, [Deep Learning for Natural Language Processing (slides and recording from NAACL 2013)](http://nlp.stanford.edu/courses/NAACL2013/)
* 2015, Yoshua Bengio et al., [Deep Learning - MIT Press book in preparation](http://www.iro.umontreal.ca/~bengioy/dlbook/)
* 2015, Richard Socher, [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)

## Lecture 1 - Introduction to Deep Learning 
*Monday, 5th October, 11am (German time zone), Room B002*

**Video:** https://youtu.be/AmG4jzmBZ88

**Slides:** [pdf](https://github.com/nreimers/deeplearning4nlp-tutorial/raw/master/2015-10_Lecture/Lecture1/2015-10-05_Deep_Learning_Intro.pdf)

**Lecture-Content:**
* Overview of the lecture
* What is Deep Learning? What is not Deep Learning?
* Fundamental terminology
* Feed Forward Neural Networks 
* Shallow vs deep neural networks

**Recommended Readings**
* From the [2009 Yoshua Bengio Book](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf):
 * *1 Introduction* - Good introduction into the terminology and the basic concept of deep learning
 * *2 Theoretical Advantages of Deep Architectures* - In case you are interested why deep architectures are better than shallow
* [Chapter 1 - Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
* [Chapter 2 - How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
* From the [2015 Yoshua Bengio Book](http://www.iro.umontreal.ca/~bengioy/dlbook/):
 * *1 Introduction* - if you are interested in the historical development of neural networks
 * *Part I: Applied Math and Machine Learning Basics* - As a refresher for linear algebra and machine learning basics (if needed)
 * *6 Feedforward Deep Networks* - Read on feedforward networks, the most simple type of neural networks


## Lecture 2 - Introduction to Theano and Lasagne 
*Monday, 12th October, 11am (German time zone), Room B002*

**Preparation before class:**
* Install Python (2.7), NumPy, SciPy and Theano. ([Installing Theano for Ubuntu](http://deeplearning.net/software/theano/install_ubuntu.html))
* Install [Lasagne](https://github.com/Lasagne/Lasagne)
* Refresh your knowledge on Python and Numpy:
  * [Python and Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/) 
  * [Python-Tutorial](http://deeplearning.net/software/theano/tutorial/python.html) and [Numpy refresher](http://deeplearning.net/software/theano/tutorial/numpy.html) from the Theano website
* **Hint:** You can install Python, Theano etc. on you local desktop machine and log into it via SSH or via [IPython Notebook](http://cs231n.github.io/ipython-tutorial/) during class


**Lecture-Content:**
* Introduction to Theano (knowledge of Python and Numpy is assumed)
* Computation graphs
* Using Theano to classify hand written digits (MNIST dataset)
* Usage of Lasagne


## Lecture 3 - Word Embeddings, Feed Forward Networks and SENNA
*Monday, 19th October, 11am (German time zone), Room B002*

**Practice-Task**:
* **Task**: Implement a Named Entity Recognizer based on the SENNA Architecture
* *A Python skeleton and a sample solution will be added to this git repository later*

## Lecture 4
*Monday, 26th October, 11am (German time zone), Room B002*

## Lecture 5
*Monday, 2nd November, 11am (German time zone), Room B002*

## Lecture 6
*Monday, 9h November, 11am (German time zone), Room B002*
