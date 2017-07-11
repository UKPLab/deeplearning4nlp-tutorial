"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from sentiment classification task (English)

It requires pre-trained word embeddings for English. It tries to download those from here:
https://www.cs.york.ac.uk/nlp/extvec/

using wget. You can also use other, pre-trained embeddings, e.g.:
- Levy & Goldberg: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
- GloVe: https://nlp.stanford.edu/projects/glove/


Code was written & tested with:
- Python 2.7 & Python 3.6

@author: Nils Reimers, www.deeplearning4nlp.com
"""
from __future__ import print_function
import numpy as np
import gzip
import os

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

#We download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
embeddingsPath = 'embeddings/wiki_extvec.gz'

#Train, Dev, and Test files
folder = 'data/'
files = [folder+'train.txt',  folder+'dev.txt', folder+'test.txt']




def createMatrices(sentences, word2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
    
    
    xMatrix = []
    unknownWordCount = 0
    wordCount = 0
    
    for sentence in sentences:
        targetWordIdx = 0
        
        sentenceWordIdx = []
        
        for word in sentence:
            wordCount += 1
            
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
                
            sentenceWordIdx.append(wordIdx)
            
        xMatrix.append(sentenceWordIdx)
       
    
    print("Unknown tokens: %.2f%%" % (unknownWordCount/(float(wordCount))*100))
    return xMatrix

def readFile(filepath):
    sentences = []    
    labels = []
    
    for line in open(filepath):   
        splits = line.split()
        label = int(splits[0])
        words = splits[1:]
        
        labels.append(label)
        sentences.append(words)
        
    print(filepath, len(sentences), "sentences")
    
    return sentences, labels

        
        



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

outputFilePath = 'pkl/data.pkl.gz'


trainDataset = readFile(files[0])
devDataset = readFile(files[1])
testDataset = readFile(files[2])


# :: Compute which words are needed for the train/dev/test set ::
words = {}
for sentences, labels in [trainDataset, devDataset, testDataset]:       
    for sentence in sentences:
        for token in sentence:
            words[token.lower()] = True


# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

# :: Downloads the embeddings from the York webserver ::
if not os.path.isfile(embeddingsPath):
    basename = os.path.basename(embeddingsPath)
    if basename == 'wiki_extvec.gz':
	       print("Start downloading word embeddings for English using wget ...")
	       #os.system("wget https://www.cs.york.ac.uk/nlp/extvec/"+basename+" -P embeddings/")
	       os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_english_embeddings/"+basename+" -P embeddings/")
    else:
        print(embeddingsPath, "does not exist. Please provide pre-trained embeddings")
        exit()
        
# :: Load the pre-trained embeddings file ::
fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
	
print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.decode("utf-8").strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)
       
        
wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))



# :: Create matrices ::
train_matrix = createMatrices(trainDataset[0], word2Idx)
dev_matrix = createMatrices(devDataset[0], word2Idx)
test_matrix = createMatrices(testDataset[0], word2Idx)


data = {
    'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
    'train': {'sentences': train_matrix, 'labels': trainDataset[1]},
    'dev':   {'sentences': dev_matrix, 'labels': devDataset[1]},
    'test':  {'sentences': test_matrix, 'labels': testDataset[1]}
    }


f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder")


        
