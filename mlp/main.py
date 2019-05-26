'''
Application classifying datasetA_3c

@author: pawel@kasprowski.pl
'''
import pandas
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing.label import LabelBinarizer

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from collections import Counter


def main():

    #load data 
    file = "datasetA_3c.csv"
    dataframe = pandas.read_csv(file)
    dataset = dataframe.values
    samples = dataset[:,1:]
    labels = dataset[:,0]
    samples = np.array(samples)
    labels = np.array(labels)
    labels = labels.astype(str)

    print("Class distribution:")
    print(Counter(labels))

### choose k best attributes
#    from sklearn.feature_selection.univariate_selection import SelectKBest
#    newSamples = SelectKBest(k=100).fit_transform(samples, labels)
#    print(newSamples.shape) 
#    samples = newSamples

### Calculate weights for unbalanced classes
#    from sklearn.utils import class_weight
#    d_class_weights = None
#    class_weights = class_weight.compute_class_weight('balanced',np.unique(labels),labels)
#    print("Class weights:")
#    print(class_weights)
#    d_class_weights = dict(enumerate(class_weights))

### Normalize samples
#    from sklearn.preprocessing.data import normalize
#    normalize(samples)


    ## convert to one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    classesNum = labels.shape[1]
    print ("Classes: {}".format(classesNum))

    trainSamples = samples
    trainLabels = labels
    testSamples = samples
    testLabels = labels
    
### Division into training and test samples
#    from sklearn.model_selection._split import train_test_split
#    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.25, random_state=42)
    
    
    model = Sequential()
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dense(classesNum, activation='softmax'))

    ## loss function depends on the number of classes!
    loss='categorical_crossentropy'    
    model.compile(loss=loss, optimizer="adam",metrics=['accuracy'])

    EPOCHS=100
    BATCH=50
    H = model.fit(trainSamples, trainLabels, batch_size=BATCH, epochs=EPOCHS
              #,class_weight=d_class_weights
              #,validation_data=(testSamples,testLabels)
              #,validation_split=0.1
              )
    mlpResults = model.predict(testSamples)


    print(confusion_matrix(testLabels.argmax(axis=1), mlpResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), mlpResults.argmax(axis=1),target_names=lb.classes_))
    mlpAcc = accuracy_score(testLabels.argmax(axis=1), mlpResults.argmax(axis=1)) 
    print("MLP Accuracy: {:.2f}".format(mlpAcc))


    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    #plt.plot(N, H.history["val_loss"], label="val_loss")
    #plt.plot(N, H.history["val_acc"], label="val_acc")
    
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

### Decision Tree model for comparison
#    from sklearn.tree import DecisionTreeClassifier
#    treemodel = DecisionTreeClassifier()
#    treemodel.fit(trainSamples, trainLabels)
#    treeResults = treemodel.predict(testSamples)    
#    print(confusion_matrix(testLabels.argmax(axis=1), treeResults.argmax(axis=1)))
#    print(classification_report(testLabels.argmax(axis=1), treeResults.argmax(axis=1)))
#    treeAcc = accuracy_score(testLabels.argmax(axis=1), treeResults.argmax(axis=1)) 
#    print("Tree Accuracy: {:.2f}".format(treeAcc))



if __name__ == "__main__":
    main()