'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Tree methods that:
- build the model using trainSamples and trainlabels
- return labels predicted by the model for testSamples

tree() - DecisionTree
flat_network() - classic Multilayer Perceptrion
cnn_network() - example of Convolutional Neural Network

@author: pawel@kasprowski.pl
'''
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

def tree(trainSamples, trainLabels, testSamples):
    
    # samples must be flat (i.e. one dimensional) for tree!
    length = trainSamples.shape[1]
    trainSamples = trainSamples.reshape(-1,length*length)
    testSamples = testSamples.reshape(-1,length*length)
    
    model = DecisionTreeClassifier()
    model.fit(trainSamples, trainLabels)    
    testResults = model.predict(testSamples)
    return testResults

def flat_network(trainSamples, trainLabels, testSamples):

    # samples must be flat (i.e. one dimensional) for MLP!
    length = trainSamples.shape[1]
    trainSamples = trainSamples.reshape(-1,length*length)
    testSamples = testSamples.reshape(-1,length*length)

    classes_number = trainLabels.shape[1]

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(classes_number, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy'])
    EPOCHS = 20
    BATCH=50
    model.fit(trainSamples, trainLabels, batch_size=BATCH, epochs=EPOCHS)
    testResults = model.predict(testSamples)
    return testResults
    
def cnn_network(trainSamples, trainLabels, testSamples):
    
    # samples must 3D for CNN!
    trainSamples = np.expand_dims(trainSamples, axis=3)
    testSamples = np.expand_dims(testSamples, axis=3)

    classes_number = trainLabels.shape[1]

    length = trainSamples.shape[1]
    inputShape = (length,length,1)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("sigmoid"))
    model.add(Dense(512))
    model.add(Activation("sigmoid"))
    model.add(Dense(classes_number))
    model.add(Activation("softmax"))

    model.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy'])
    EPOCHS = 10
    BATCH=50
    model.fit(trainSamples, trainLabels, batch_size=BATCH, epochs=EPOCHS)
    testResults = model.predict(testSamples)
    return testResults
