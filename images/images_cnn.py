'''
Classification of eye tracking data from /data folder into one of four categories
1) Loading data
2) Extracting sequences
3) Decision Tree classification
4) CNN classification

@author: pawel@kasprowski.pl
'''
import os
import numpy as np

from sklearn.metrics.classification import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.tree.tree import DecisionTreeClassifier
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Activation, Flatten, Dropout, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers.normalization import BatchNormalization



def load_files(indir,sequence_len=200):
    samples = []
    names = []
    labels = []
    for file in os.listdir(indir):
        sample = np.genfromtxt(os.path.join(indir, file), delimiter='\t')
        sample = sample[:,2:4] ##omit irrelevant columns
        
        ## convert position to velocity
        vsample = np.zeros((sample.shape[0],sample.shape[1]))
        for i in range(1,sample.shape[0]):
            vsample[i] = sample[i]-sample[i-1]
        sample = vsample    
        samples.append(sample)
        names.append(file)
        labels.append(file[0:3])
    samples = np.array(samples)
    labels = np.array(labels)
    return samples,labels,names

def make_sequences(samples, labels, sequence_dim = 100, sequence_lag = 1, sequence_attributes = 2):
    nsamples = []
    nlabels = []
    for s in range(samples.shape[0]):
    #for sample in samples:
        sample = samples[s]
        for i in range(0,len(sample)-sequence_dim,sequence_lag):
            nsample = np.zeros((sequence_dim,sequence_attributes))
            for j in range(i,i+sequence_dim):
                for k in range(sequence_attributes):
                    nsample[j-i,k] = sample[j,k]
            nsamples.append(nsample)
            nlabels.append(labels[s])
        
    samples = np.array(nsamples, dtype="float")
    labels = np.array(nlabels)
    return samples,labels

def main():
    print("Loading samples and labels")
    samples,labels,_ = load_files("data",sequence_len=200)
    print("Loaded {} samples".format(samples.shape[0]))
    
    sequence_dim = 100
    print("Converting to sequences of length {}".format(sequence_dim))
    samples, labels = make_sequences(samples, labels, sequence_dim)
    
    print("Number of samples from sequences: {}".format(samples.shape[0]))
        
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # flattened samples for Decision Tree
    flatSamples = samples.reshape(samples.shape[0],-1) #tree!
    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(flatSamples, labels, test_size=0.25, random_state=42)

    print("="*20)
    print("Building DecisionTree model")
    model = DecisionTreeClassifier()
    model.fit(trainSamples, trainLabels)    
    treeResults = model.predict(testSamples)
    print(confusion_matrix(testLabels.argmax(axis=1), treeResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), treeResults.argmax(axis=1)))
    treeAcc = accuracy_score(testLabels.argmax(axis=1), treeResults.argmax(axis=1)) 
    print("Accuracy Tree: {:.2f}".format(treeAcc))
    print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(testLabels.argmax(axis=1), treeResults.argmax(axis=1))))

    print("="*20)
    print("Building CNN model")

    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.25, random_state=42)
    inputShape = (samples.shape[1],samples.shape[2])
    model = Sequential()
    model.add(Conv1D(32, 10, padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D(64, 10, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 10, padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Flatten(input_shape=inputShape))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
    
    EPOCHS=10
    BATCH=128
    model.fit(trainSamples, trainLabels, batch_size=BATCH, epochs=EPOCHS
              ,validation_data=(testSamples,testLabels)
              )
    
    cnnResults = model.predict(testSamples)
    
    print(confusion_matrix(testLabels.argmax(axis=1), cnnResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), cnnResults.argmax(axis=1),target_names=lb.classes_))
    print("CNN Accuracy: {:.2f}".format(accuracy_score(testLabels.argmax(axis=1), cnnResults.argmax(axis=1))))
    print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(testLabels.argmax(axis=1), cnnResults.argmax(axis=1))))
    input("")

main()