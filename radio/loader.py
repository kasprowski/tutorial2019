'''


@author: pawel@kasprowski.pl
'''
import numpy as np
import pandas
import cv2
import os
from keras_preprocessing.sequence import pad_sequences


def load_csv(file):
    dataframe = pandas.read_csv(file)
    dataset = dataframe.values

    samples = dataset[:,4:]
    labels = dataset[:,0]
    names = []
    for i in range(len(dataset)):
        names.append(str(dataset[i,0])+"_"+str(dataset[i,1])+"_"+str(dataset[i,2])+"_"+str(dataset[i,3]))
    samples = np.array(samples)
    labels = np.array(labels)
    names = np.array(names)
    return samples,labels,names

def load_img(indir):
    samples = []
    labels = []
    names = []
    for file in os.listdir(indir):
        sample = cv2.imread("{}/{}".format(indir,file))
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        sample = cv2.resize(sample,(192,108))
        samples.append(sample)
        labels.append(file[:1])
        names.append(file[:-4])
    samples = np.array(samples)
    labels = np.array(labels)
    return samples,labels,names

def load_files(indir,sequence_len=1000):
    samples = []
    names = []
    labels = []
    for file in os.listdir(indir):
        sample = np.genfromtxt(os.path.join(indir, file), delimiter=',', skip_header=1)
        if sample.shape[0]>100:
            samples.append(sample)
            names.append(file)
            labels.append(file[0])
    samples = np.array(samples)
    labels = np.array(labels)
    samples = pad_sequences(samples,sequence_len)
    samples = samples[:,:,1:] ##omit timestamp
    return samples,labels,names
