'''
Created on 05.04.2019

@author: win
'''
import random
import matplotlib.pyplot as plt

import cv2
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


from models import cnn_network, flat_network, tree

import numpy as np


length = 100 #size of images
size = 100 #number of samplesIMG per class



def printResults(testLabels,testResults):
    correct = 0
    size = len(testLabels)
    lbs = np.zeros(4)
    for i in range(size):
        print("{} - {} : {}".format(testLabels[i],testResults[i],(testLabels[i].argmax(axis=0)==testResults[i].argmax(axis=0))))
        if testLabels[i].argmax(axis=0)==testResults[i].argmax(axis=0):
            correct+=1
        lbs[testResults[i].argmax(axis=0)]+=1

    print("correct: {} accuracy: {}".format(correct,correct/size))
    print(lbs)
    return correct/size

###################################################



def prepare_samples():
    samplesIMG = []
    labels = []
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(100):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=255
        #samplesIMG.append(sample.flatten())
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("h.jpg",sample)
    #    cv2.imwrite("h{}.jpg".format(i),sample)
    
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(100):
            x = random.randrange(2,97)
            y = random.randrange(2,97)
            sample[x:x+1,y:y+6]=255
            #sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("v.jpg",sample)

    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(100):
            x = random.randrange(2,98)
            y = random.randrange(2,98)
            sample[x,y]=255
            sample[x-1,y-1]=255
            sample[x-1,y-1]=255
            sample[x+1,y+1]=255
            sample[x+2,y+2]=255
            #sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(2)
        if i==0: cv2.imwrite("b.jpg",sample)

    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(100):
            x = random.randrange(2,98)
            y = random.randrange(2,98)
            sample[x,y]=255
            sample[x+1,y-1]=255
            sample[x+1,y-1]=255
            sample[x-1,y+1]=255
            sample[x-2,y+2]=255
        samplesIMG.append(sample)
        labels.append(3)
        if i==0: cv2.imwrite("s.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    return samplesIMG,labels

def plot_accuracy(values):
    hfig = cv2.imread('h.jpg')
    vfig = cv2.imread('v.jpg')
    bfig = cv2.imread('b.jpg')
    sfig = cv2.imread('s.jpg')
    fig, axs = plt.subplots(4, 3, figsize=(5, 5))
    axs[0,0].imshow(hfig)
    axs[1,0].imshow(vfig)
    axs[2,0].imshow(bfig)
    axs[3,0].imshow(sfig)

    axs[0,1].remove()
    axs[1,1].remove()
    axs[2,1].remove()
    axs[3,1].remove()
    
    axs[0,2].remove()
    axs[1,2].remove()
    axs[2,2].remove()
    axs[3,2].remove()

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
    samplesIMG,labels = prepare_samples()
    
    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samplesIMG, labels, test_size=0.25, random_state=42)
    
    print("TREE")
    testResults = tree(trainSamples,trainLabels,testSamples)
    accTree = printResults(testLabels, testResults)
    
    print("FLAT")
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