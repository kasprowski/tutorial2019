'''
Created on 20.05.2019

@author: win
'''
import numpy as np
import random
import cv2
import tensorflow as tf

from sklearn.preprocessing.label import LabelBinarizer


length = 100 #size of images
size = 100 #number of samples for each class


def prepare_samples_areas():
    samplesIMG = []
    labels = []
    dashes = 50 #number of dashes
    
    #images with horizontal lines
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(dashes):
            x = random.randrange(0,length/2)
            y = random.randrange(0,length/2)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("v.jpg",sample)
    
    #images with vertical lines
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(dashes):
            x = random.randrange(length/2,length)
            y = random.randrange(length/2,length)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("h.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels,2)
    return samplesIMG,labels

def prepare_samples_colors():
    samplesIMG = []
    labels = []
    dashes = 50 #number of dashes
    
    #images with horizontal lines
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("v.jpg",sample)
    
    #images with vertical lines
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=155
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("h.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels,2)
    return samplesIMG,labels


def prepare_samples_direction():
    samplesIMG = []
    labels = []
    dashes = 100 #number of dashes
    
    #images with horizontal lines
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("h.jpg",sample)
    
    #images with vertical lines
    for i in range(size):
        sample = np.zeros((length,length))
        for j in range(100):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+1,y:y+6]=255
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("v.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels,2)
    return samplesIMG,labels
