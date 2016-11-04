"""
Create Train and Test Files for the SemEval 2010 Task 8 experiment
"""
import nltk

trainFile = 'corpus/SemEval2010_task8_training/TRAIN_FILE.TXT'
testFile = 'corpus/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'




def createFile(filepath, outputpath):
    fOut = open(outputpath, 'w')
    lines = [line.strip() for line in open(filepath)]
    for idx in xrange(0, len(lines), 4):
        sentence = lines[idx].split("\t")[1][1:-1]
        label = lines[idx+1]
        
        sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")
        tokens = nltk.word_tokenize(sentence)
        #print tokens
        tokens.remove('_/e1_')    
        tokens.remove('_/e2_')
        
        e1 = tokens.index("_e1_")
        del tokens[e1]
        
        e2 = tokens.index("_e2_")
        del tokens[e2]
        
        #print tokens
        #print tokens[e1], "<->", tokens[e2]
    
        fOut.write("\t".join([label, str(e1), str(e2), " ".join(tokens)]))
        fOut.write("\n")
    fOut.close()
    
createFile(trainFile, "files/train.txt")
createFile(testFile, "files/test.txt")

print "Train / Test file created"