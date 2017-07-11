from random import shuffle


def writeFile(filename, sentences):
    with open(filename, 'w') as fOut:
        for line in sentences:
            fOut.write(line)
            fOut.write("\n")
            
negSentences = []
posSentences = []

for line in open('negative.txt'):
    line = "0\t"+line.strip()
    negSentences.append(line)
    
for line in open('positive.txt'):
    line = "1\t"+line.strip()
    posSentences.append(line)
    
    
shuffle(negSentences)
shuffle(posSentences)
    
n = len(negSentences)    
trainSplit = (0, int(0.5*n))
devSplit = (trainSplit[1], trainSplit[1]+int(0.25*n))
testSplit = (devSplit[1], n)

train = negSentences[trainSplit[0]:trainSplit[1]] + posSentences[trainSplit[0]:trainSplit[1]]
dev = negSentences[devSplit[0]:devSplit[1]] + posSentences[devSplit[0]:devSplit[1]]
test = negSentences[testSplit[0]:testSplit[1]] + posSentences[testSplit[0]:testSplit[1]]

writeFile('train.txt', train)
writeFile('dev.txt', dev)
writeFile('test.txt', test)
