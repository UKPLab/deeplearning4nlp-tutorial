"""
createNumpyArrayWithCasing returns the X-Matrix for the word embeddings as well as for the case information
and the Y-vector with the labels
@author: Nils Reimers
"""
import numpy as np
from GermEvalReader import normalizeWord

def createNumpyArrayWithCasing(sentences, windowsize, word2Idx, label2Idx, caseLookup):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']    
    
    
    
    xMatrix = []
    caseMatrix = []
    yVector = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        targetWordIdx = 0
        
        for targetWordIdx in xrange(len(sentence)):
            
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []    
            caseIndices = []
            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    caseIndices.append(caseLookup['PADDING'])
                    continue
                
                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()] 
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)] 
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                
                
                wordIndices.append(wordIdx)
                caseIndices.append(getCasing(word, caseLookup))
                
            #Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]
            
            #Get the casing            
            xMatrix.append(wordIndices)
            caseMatrix.append(caseIndices)
            yVector.append(labelIdx)
    
    
    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return (np.asarray(xMatrix), np.asarray(caseMatrix), np.asarray(yVector))

def getCasing(word, caseLookup):   
    casing = 'other'
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    if word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
   
    return caseLookup[casing]