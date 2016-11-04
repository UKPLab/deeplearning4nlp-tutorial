"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Non-other Macro-Averaged F1:  0.7612

Code was tested with:
- Theano 0.8.2
- Keras 1.1.0
- Python 2.7
"""
import theano
import numpy as np
np.random.seed(1337)  # for reproducibility

import cPickle as pkl
import gzip
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, GlobalMaxPooling1D


from keras.utils import np_utils



batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50

print "Load dataset"
f = gzip.open('pkl/sem-relations.pkl', 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)
test_y_cat = np_utils.to_categorical(yTest, n_out)

print "sentenceTrain: ", sentenceTrain.shape
print "positionTrain1: ", positionTrain1.shape
print "yTrain: ", yTrain.shape




print "sentenceTest: ", sentenceTest.shape
print "positionTest1: ", positionTest1.shape
print "yTest: ", yTest.shape


f = gzip.open('pkl/embeddings.pkl', 'rb')
embeddings = pkl.load(f)
f.close()

print "Embeddings: ",embeddings.shape

distanceModel1 = Sequential()
distanceModel1.add(Embedding(max_position, position_dims, input_length=positionTrain1.shape[1]))

distanceModel2 = Sequential()
distanceModel2.add(Embedding(max_position, position_dims, input_length=positionTrain2.shape[1]))



wordModel = Sequential()
wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False))


convModel = Sequential()
convModel.add(Merge([wordModel, distanceModel1, distanceModel2], mode='concat'))




convModel.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='tanh',
                        subsample_length=1))
# we use standard max over time pooling
convModel.add(GlobalMaxPooling1D())


model = convModel

#model.add(Dropout(0.25))
#model.add(Dense(hidden_dims,  activation='tanh', W_regularizer=keras.regularizers.l2(0.01)))
model.add(Dropout(0.5))


model.add(Dense(n_out, activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])

model.summary()
print "Start training"



max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in xrange(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

for epoch in xrange(nb_epoch):   
    
    model.fit([sentenceTrain, positionTrain1, positionTrain2], train_y_cat, batch_size=batch_size, verbose=True,nb_epoch=1)
    #model.fit([sentenceTest, positionTest1, positionTest2], test_y_cat, batch_size=batch_size, verbose=True,nb_epoch=1)
    
    pred_test = model.predict_classes([sentenceTest, positionTest1, positionTest2], verbose=False)
    
           
    
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(yTest)
   
      
      
   
    acc =  np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)
    
    

    f1Sum = 0
    f1Count = 0
    for targetLabel in xrange(1, max(yTest)):        
        prec = getPrecision(pred_test, yTest, targetLabel)
        rec = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
        f1Sum += f1
        f1Count +=1    
        
        
    macroF1 = f1Sum / float(f1Count)
    
    max_f1 = max(max_f1, macroF1)

    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)
    
    
  



print "DONE"
