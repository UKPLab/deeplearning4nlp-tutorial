"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from Universal Dependencies, POS-tagging (English)

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

#At which column position is the token and the tag, starting at 0
tokenPosition=0
tagPosition=1

#Size of the context window
window_size = 3

def createMatrices(sentences, windowsize, word2Idx, label2Idx, case2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
    
    
    xMatrix = []
    caseMatrix = []
    yVector = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        targetWordIdx = 0
        
        for targetWordIdx in range(len(sentence)):
            
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []    
            caseIndices = []
            for wordPosition in range(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    caseIndices.append(case2Idx['PADDING_TOKEN'])
                    continue
                
                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]                 
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                
                
                wordIndices.append(wordIdx)
                caseIndices.append(getCasing(word, case2Idx))
                
            #Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]
            
            #Get the casing            
            xMatrix.append(wordIndices)
            caseMatrix.append(caseIndices)
            yVector.append(labelIdx)
    
    
    print("Unknown tokens: %.2f%%" % (unknownWordCount/(float(wordCount))*100))
    return (np.asarray(xMatrix), np.asarray(caseMatrix), np.asarray(yVector))

def readFile(filepath, tokenPosition, tagPosition):
    sentences = []
    sentence = []
    
    for line in open(filepath):
        line = line.strip()
        
        if len(line) == 0 or line[0] == '#':
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        sentence.append([splits[tokenPosition], splits[tagPosition]])
    
    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
        
    print(filepath, len(sentences), "sentences")
    return sentences

def getCasing(word, caseLookup):   
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
   
    return caseLookup[casing]
           
        



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

outputFilePath = 'pkl/data.pkl.gz'
embeddingsPklPath = 'pkl/embeddings.pkl.gz'


trainSentences = readFile(files[0], tokenPosition, tagPosition)
devSentences = readFile(files[1], tokenPosition, tagPosition)
testSentences = readFile(files[2], tokenPosition, tagPosition)

#Mapping of the labels to integers
labelSet = set()
words = {}

for dataset in [trainSentences, devSentences, testSentences]:
    for sentence in dataset:
        for token, label in sentence:
            labelSet.add(label)
            words[token.lower()] = True
            

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)


# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
        
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []


# :: Downloads the embeddings from the York webserver ::
if not os.path.isfile(embeddingsPath):
    basename = os.path.basename(embeddingsPath)
    if basename == 'wiki_extvec.gz':
	       print("Start downloading word embeddings for English using wget ...")
	       os.system("wget https://www.cs.york.ac.uk/nlp/extvec/"+basename+" -P embeddings/")
    else:
        print(embeddingsPath, "does not exist. Please provide pre-trained embeddings")
        exit()
     
# :: Load the pre-trained embeddings file ::   
fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
    
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

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))

embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
              'caseEmbeddings': caseEmbeddings, 'case2Idx': case2Idx,
              'label2Idx': label2Idx}

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()

# :: Create matrices ::
train_set = createMatrices(trainSentences, window_size, word2Idx,  label2Idx, case2Idx)
dev_set = createMatrices(devSentences, window_size, word2Idx, label2Idx, case2Idx)
test_set = createMatrices(testSentences, window_size, word2Idx, label2Idx, case2Idx)



f = gzip.open(outputFilePath, 'wb')
pkl.dump(train_set, f, -1)
pkl.dump(dev_set, f, -1)
pkl.dump(test_set, f, -1)
f.close()

print("Data stored in pkl folder")


        
