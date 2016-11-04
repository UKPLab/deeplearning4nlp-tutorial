#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Creates from a .vocab-file a subcorpus, only containing words+embeddings were are interested in

Run by: create_sub_corpus.py wordfile.txt embeddings.vocab

@author: Nils Reimers
"""
import sys
import re
from unidecode import unidecode

wordsFile = sys.argv[1]
embeddingsFile = sys.argv[2]
subFile = embeddingsFile+'_sub'

def multiple_replacer( key_values):
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


words = {}

#Read in words
for line in open(wordsFile, 'r'):
    word = line.strip();
    words[word] = True
    words[word.lower()] = True
    words[normalizeWord(word)] = True
    


#Read in embeddings
fOut = open(subFile, 'w')
for line in open(embeddingsFile, 'r'):
    splits = line.strip().split(' ',1)    
    word = splits[0]
    
 
    if word in words:        
        fOut.write(line)
        
    
    
fOut.close()
        
print "Done"


