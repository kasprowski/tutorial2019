'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Test of classification models using two types of generated images
Uses: data.py to prepare data, models.py to retrieve models
 
@author: pawel@kasprowski.pl
'''

import cv2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

from data import prepare_samples_areas, prepare_samples_direction, prepare_samples_colors
import matplotlib.pyplot as plt
from models import cnn_network, flat_network, tree


def printResults(testLabels,testResults):
    print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
    print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
    return accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))

###################################################

def plot_accuracy(values):
    hfig = cv2.imread('h.jpg')
    vfig = cv2.imread('v.jpg')
    fig, axs = plt.subplots(2, 3, figsize=(5, 5))
    axs[0,0].imshow(hfig)
    axs[1,0].imshow(vfig)
    axs[0,1].remove()
    axs[1,1].remove()
    axs[0,2].remove()
    axs[1,2].remove()
    gs = axs[0,1].get_gridspec()
    labels = ["Random Forest","Flat NN","CNN"]
    p = fig.add_subplot(gs[:, 1:])
    p.bar(labels, values)
    p.set_xlabel('Model')
    p.set_ylabel('Accuracy')
    p.set_xticklabels(labels)
    p.set_title('Accuracy for given image types')
    fig.tight_layout()
    plt.show()


def main():
    #samples,labels = prepare_samples_areas()
    samples,labels = prepare_samples_direction()
    #samples,labels = prepare_samples_colors()

    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.25, random_state=42)
    
    print("TREE")
    testResults = tree(trainSamples,trainLabels,testSamples)
    accTree = printResults(testLabels, testResults)
    
    print("MLP - FLAT")
    testResults = flat_network(trainSamples,trainLabels,testSamples)
    accFlat = printResults(testLabels, testResults)
        
    print("CNN")
    testResults = cnn_network(trainSamples,trainLabels,testSamples)
    accCnn = printResults(testLabels, testResults)

    print("Accuracy TREE: {}".format(accTree))
    print("Accuracy FLAT: {}".format(accFlat))
    print("Accuracy CNN: {}".format(accCnn))
    plot_accuracy((accTree,accFlat,accCnn))

if __name__ == "__main__":
    main()    