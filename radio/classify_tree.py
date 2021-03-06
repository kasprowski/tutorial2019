'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Classification of "radio" samples using DecisionTree
Script loads data from /radio folder and classifies all samples as:
L - layman
R - resident
S - radiology specialist  

@author: pawel@kasprowski.pl
'''
import loader

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, cohen_kappa_score
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
    print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(testLabels.argmax(axis=1), treeResults.argmax(axis=1))))
    input("")

if __name__ == "__main__":
    main()