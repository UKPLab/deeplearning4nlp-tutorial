# -*- coding: utf-8 -*-
"""
readFile: Reads in a file from the GermEval 2014 dataset and returns the sentences together with the gold labels
createDataset: Maps the tokens in the sentence as well as the labels to the corresponding word indicies. The format of the return looks like this:
    [ [wordIndices1, caseIndices1, labelIndices1], [wordIndices2, caseIndices2, labelIndices2] ...]
    each entry corresponding to a sentence in the dataset.
     
@author: Nils Reimers
"""
import numpy as np
import re
from unidecode import unidecode

def createDataset(sentences, word2Idx, label2Idx, caseLookup):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']    
    
    data = []
    
    for sentence in sentences:
        targetWordIdx = 0
        
        wordIndices = []
        caseIndices = []
        labelIndices = []
        
        for wordPosition in xrange(len(sentence)):
            word, label = sentence[wordPosition]
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()] 
            elif normalizeWord(word) in word2Idx:
                wordIdx = word2Idx[normalizeWord(word)] 
            else:
                wordIdx = unknownIdx
                
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, caseLookup))
            labelIndices.append(label2Idx[label])
            
            
        
        data.append([wordIndices, caseIndices, labelIndices])
    return data

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
        
    
    if len(sentence) > 0:
        sentences.append(sentence)
    
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