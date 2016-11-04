"""
The file preprocesses the files/train.txt and files/test.txt files.

I requires the dependency based embeddings by Levy et al.. Download them from his website and change 
the embeddingsPath variable in the script to point to the unzipped deps.words file.
"""

import numpy as np
import cPickle as pkl
from nltk import FreqDist
import gzip

outputFilePath = 'pkl/sem-relations.pkl.gz'
embeddingsPklPath = 'pkl/embeddings.pkl.gz'

#Download from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ the deps.words.bz file
#and unzip it. Change the path here to the correct path for the embeddings file
embeddingsPath = '/home/likewise-open/UKP/reimers/NLP/Models/Word Embeddings/English/levy_dependency_based.deps.words'


folder = 'files/'
files = [folder+'train.txt', folder+'test.txt']

#Mapping of the labels to integers
labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}




words = {}
maxSentenceLen = [0,0,0]
labelsDistribution = FreqDist()

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in xrange(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)



def createMatrices(file, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []
    
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3]
        tokens = sentence.split(" ")
        
        labelsDistribution[label] += 1
      
        
        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)
        
        for idx in xrange(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)
            
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']
                
            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']
            
        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)
        
        labels.append(labelsMapping[label])
        

    
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
        
        
        
 
def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN"]



for fileIdx in xrange(len(files)):
    file = files[fileIdx]
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        
        
        sentence = splits[3]        
        tokens = sentence.split(" ")
        maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True
            

print "Max Sentence Lengths: ",maxSentenceLen
        
# :: Read in word embeddings ::


word2Idx = {}
embeddings = []



for line in open(embeddingsPath):
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        embeddings.append(vector)
        
        word2Idx["UNKNOWN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        embeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        embeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
embeddings = np.array(embeddings)

print "Embeddings shape: ", embeddings.shape
print "Len words: ", len(words)

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()

# :: Create token matrix ::
train_set = createMatrices(files[0], word2Idx, max(maxSentenceLen))
test_set = createMatrices(files[1], word2Idx, max(maxSentenceLen))



f = gzip.open(outputFilePath, 'wb')
pkl.dump(train_set, f, -1)
pkl.dump(test_set, f, -1)
f.close()



print "Data stored in pkl folder"

for label, freq in labelsDistribution.most_common(100):
    print "%s : %f%%" % (label, 100*freq / float(labelsDistribution.N()))
        
        