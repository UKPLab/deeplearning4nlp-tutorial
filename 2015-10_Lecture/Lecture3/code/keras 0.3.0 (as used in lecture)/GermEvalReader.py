# -*- coding: utf-8 -*-

import re
from unidecode import unidecode
import numpy as np

"""
Functions to read in the files from the GermEval contest, 
create suitable numpy matrices for train/dev/test

@author: Nils Reimers
"""


def readFile(filepath):
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
        sentence.append([splits[1], splits[2]])
    
    return sentences

def multiple_replacer(key_values):
    #replace_dict = dict(key_values)
    replace_dict = key_values
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values.iteritems()]), re.M)
    return lambda string: pattern.sub(replacement_function, string)
    

def multiple_replace(string, key_values):
    return multiple_replacer(key_values)(string)

def normalizeWord(line):         
    line = unicode(line, "utf-8") #Convert to UTF8
    line = line.replace(u"„", u"\"")
   
    line = line.lower(); #To lower case
     
    #Replace all special charaters with the ASCII corresponding, but keep Umlaute
    #Requires that the text is in lowercase before
    replacements = dict(((u"ß", "SZ"), (u"ä", "AE"), (u"ü", "UE"), (u"ö", "OE")))
    replacementsInv = dict(zip(replacements.values(),replacements.keys()))
    line = multiple_replace(line, replacements)
    line = unidecode(line)
    line = multiple_replace(line, replacementsInv)
     
    line = line.lower() #Unidecode might have replace some characters, like € to upper case EUR
     
    line = re.sub("([0-9][0-9.,]*)", '0', line) #Replace digits by NUMBER        

   
    return line.strip();
        
def createNumpyArray(sentences, windowsize, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']    
    
    xMatrix = []
    yVector = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        targetWordIdx = 0
        
        for targetWordIdx in xrange(len(sentence)):
            
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []    
            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
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
                
            #Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]
            
            xMatrix.append(wordIndices)
            yVector.append(labelIdx)
    
    
    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return (np.asarray(xMatrix, dtype='int32'), np.asarray(yVector, dtype='int32'))



        
    
