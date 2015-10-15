"""
Reads in our files and outputs all words to a vocabulary file

@author: Nils Reimers
"""

filenames = ['data/NER-de-test.tsv', 'data/NER-de-dev.tsv', 'data/NER-de-train.tsv']
outputfile = 'vocabulary.txt'

words = set()

for filename in filenames:
    for line in open(filename): 
        line = line.strip()
        
        if len(line) == 0 or line[0] == '#':
            continue
        
        splits = line.split('\t')
        
        words.add(splits[1])
    
fOut = open(outputfile, 'w')    
for word in sorted(words):
    fOut.write(word+'\n')
    
print "Done, words exported"