'''
Classification of radio samples

@author: pawel@kasprowski.pl
'''
import os
import numpy as np
import loader

from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

def main():
    samples,labels,_ = loader.load_files("radio",700)

    print("shape = {}".format(samples.shape))
    
    #flatten
    samples = samples.reshape(-1,samples.shape[1]*samples.shape[2])

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    classesNum = labels.shape[1]
    print ("Classes: {}".format(classesNum))

 
    #split to training and test
    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.25, random_state=42)

    
    model = DecisionTreeClassifier()
    model.fit(trainSamples, trainLabels)    
    treeResults = model.predict(testSamples)
    print(confusion_matrix(testLabels.argmax(axis=1), treeResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), treeResults.argmax(axis=1)))
    treeAcc = accuracy_score(testLabels.argmax(axis=1), treeResults.argmax(axis=1)) 
    print("Accuracy Tree: {:.2f}".format(treeAcc))


if __name__ == "__main__":
    main()