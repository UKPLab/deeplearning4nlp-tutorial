# Deep Learning for NLP - Lecture October 2015
**This site can be access by the URL: www.deeplearning4nlp.com**


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
* 2015, Yoav Goldberg, [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)

## Lecture 1 - Introduction to Deep Learning 
*Monday, 5th October, 11am, Room B002*

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
*Monday, 12th October, 11am, Room B002*

**Preparation before class:**
* Install Python (2.7), NumPy, SciPy and Theano. ([Installing Theano for Ubuntu](http://deeplearning.net/software/theano/install_ubuntu.html))
* ~~Install [Lasagne](https://github.com/Lasagne/Lasagne)~~
 * Please use [Keras](http://keras.io), its much faster than Lasagne 
* Refresh your knowledge on Python and Numpy:
  * [Python and Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/) 
  * [Python-Tutorial](http://deeplearning.net/software/theano/tutorial/python.html) and [Numpy refresher](http://deeplearning.net/software/theano/tutorial/numpy.html) from the Theano website
* **Hint:** You can install Python, Theano etc. on you local desktop machine and log into it via SSH or via [IPython Notebook](http://cs231n.github.io/ipython-tutorial/) during class

**Slides:** [pdf](https://github.com/nreimers/deeplearning4nlp-tutorial/raw/master/2015-10_Lecture/Lecture2/2015-10-12_Theano_Introduction.pdf)

**Code:** [/Lecture2/code](https://github.com/nreimers/deeplearning4nlp-tutorial/tree/master/2015-10_Lecture/Lecture2/code)
* The code uses Python 2.7. With Python 3, you might need to change the syntax accordingly

**Video:** https://youtu.be/BCwBl_55n7s

## Lecture 3 - Word Embeddings and Deep Feed Forward Networks
*Monday, 19th October, 11am, Room B002*

**Preparation before class:**
* Install [Keras](http://keras.io)
* Know the theory of word embeddings & word2vec
* Watch from the [CS224d Stanford Class](http://cs224d.stanford.edu/syllabus.html) the following videos:
  * [Lecture 2](https://www.youtube.com/watch?v=T8tQZChniMk)
    * 00:00 - 21:30 - Introduction to word vectors via SVD 
    * 21:30 - 28:00 - Hacks for word vector learning 
    * 28:00 - 01:01:00- Problems with SVD, Introduction to word2vec (**please watch at least this part**)
      * From minute 38 to 53 he derives the optimization function of word2vec, feel free to skip this part
    * 1:01:00 - 1:13:00 - LSA vs. Skip-Gram vs. CBOW. Introduction of Glove 
  * [Lecture 3](https://www.youtube.com/watch?v=T1j2Q9_FgTM) 
    * 00:00 - 13:00 - How is word2vec trained, how are the word embeddings updated (**please watch at least this part**)
    * 13:00 - 20:00 - What is Skip-Gram, Negative Sampeling, CBOW (**please watch at least this part**)
    * 20:00 - 28:00 - How to evaluate word embeddings 
    * 28:00 - 36:00 - How to improve the quality of word embeddings 
    * 36:00 - 38:00 - Intrinsic evaluation of word embeddings
    * 38:00 - 41:00 - How to deal with ambiguous words? 
    * 41:00 - 42:00 - Intrinsic evaluation of word embeddings 
    * 42:00 - 50:45 - Using word embeddings and softmax for classification
    * 50:45 - 55:00 - Cross Entropy error
    * 55:00 - 1:08:00 - Should word embeddings be updated during classification? 
* The theory will **not** be introduced in class. But if you have questions regarding the theory / the videos, please ask them. We will discuss your questions / the videos in the beginning of Lecture 3 
* Get familiar with Theano and Lasagne, do the exercises from Lecture 2
 
**Slides** [pdf](https://github.com/nreimers/deeplearning4nlp-tutorial/raw/master/2015-10_Lecture/Lecture3/2015-10-19_Lecture3.pdf)

**Video**: https://youtu.be/MXHyIpv6RIg
* There are a lot of parts where I use flipcharts and neither the audio nor the video captures this. Basically I present the Collobert et al., NLP almost from scratch, approach. An illustration of this drawing can be found [here](https://www.werc.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/2014-10-14_LKE_Tutorial_on_Deep_Learning.pdf) slides 41-43 (SENNA, Window Approach, Sentence Approach)

**Exercises**:
 * Try different hyperparameters, e.g. window size, number of hidden units, optimization function, activation functions
 * Take the NER Keras implementation (Lecture3/code/NER_Keras.py) and extend it with a Casing feature, i.e. the information if a word is all uppercase, all lowercase or initial uppercase. Hint: Your network needs two inputs, one for the word indices, one for the chasing information. 

## Lecture 4 - Autoencoders, Recursive Neural Networks, Dropout
*Monday, 26th October, 11am, Room B002*

**Recommended Readings before class**:
* Autoencoders:
 * http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
 * http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders
 * http://ufldl.stanford.edu/wiki/index.php/Fine-tuning_Stacked_AEs
* Recursive Neural Networks:
 * [Socher et al., 2011, Semi-supervised recurisve autoencoders for predicting sentiment distributions](http://dl.acm.org/citation.cfm?id=2145450) 
 * [Socher et al., 2013, Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank](http://nlp.stanford.edu/%7Esocherr/EMNLP2013_RNTN.pdf)

**Slides:** [pdf](https://github.com/nreimers/deeplearning4nlp-tutorial/blob/master/2015-10_Lecture/Lecture4/2015-10-26_Autoencoders_Recursive_NN.pdf)

**Video:** https://youtu.be/FSKag11y8yI

**Exercise:** Have a look in the code directory. There you can find an example on the Brown corpus performing genre classifcation, one example on the 20 newsgroup dataset on topic classification and one example on autoencoders for the MNIST dataset.

## Lecture 5 - Convolutional Neural Networks
*Monday, 2nd November, 11am, Room B002*

**Recommended Readings**:
 * [CS224d - Lecture 13](https://www.youtube.com/watch?v=EevTPpQvxiU)
 * [Kim, 2014, Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)


## Lecture 6 - Recurrent models and LSTM-Model
*Monday, 9h November, 11am, Room B002*
