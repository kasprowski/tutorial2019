'''
RNN network trained to predict next gaze based on the previous 20

@author: pawel@ksprowski.pl
'''

import os
import cv2
import numpy as np

from sklearn.model_selection._split import train_test_split
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers.core import Dense


def load_files(indir):
    samples = []
    for file in os.listdir(indir):
        samples.append(np.genfromtxt(os.path.join(indir, file), delimiter=','))
    samples = np.array(samples)
    return samples

# Turns one sample of length N into N-sequence_dim samples of length sequence_dim
# (only if sequence_lag==1!)
def make_sequences(samples, sequence_dim = 20, sequence_lag = 1):
    nsamples = []
    nlabels = []
    for sample in samples:
        for i in range(0,len(sample)-sequence_dim-1,sequence_lag):
            nsample = np.zeros((sequence_dim,2))
            for j in range(i,i+sequence_dim):
                nsample[j-i,0] = sample[j,1]
                nsample[j-i,1] = sample[j,2]
            nsamples.append(nsample)
            nlabels.append(sample[i+sequence_dim,1:])
    samples = np.array(nsamples, dtype="float")
    labels = np.array(nlabels)
    return samples,labels

#############################################################################33
def main():
    samples = load_files("data")
    
    sequence_dim=20
    sequence_lag=1
    
    samples, labels = make_sequences(samples, sequence_dim, sequence_lag)
    
    model = Sequential()
    model.add(LSTM(128,input_shape=(sequence_dim,2),return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(64))
    model.add(Dense(2))
    
    print(model.summary())
    
    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.15, random_state=42)
        
    imname = "animal-11"    
    image = cv2.imread("img/{}.jpg".format(imname))
    # create ground truth image with all train gazes
    for j in range(len(trainLabels)):
        s = trainLabels[j]
        cv2.circle(image,(int(s[0]),int(s[1])),10,(255,0,0),3)
    cv2.imwrite("img/{}_truth.jpg".format(imname),image)
    
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mae"])
    
    EPOCHS = 30
    for e in range(EPOCHS):
        print("="*50)
        print("Iteration: {}".format(e))
        model.fit(trainSamples, trainLabels, validation_data=(testSamples, testLabels), epochs=1
                  , batch_size=128, verbose=1)
        
        predictions = model.predict(testSamples)
        image = cv2.imread("img/{}.jpg".format(imname))
        cv2.line(image,(0,0),(200,200),(255,255,255),2)
        for p in predictions:    
            cv2.circle(image,(int(p[0]),int(p[1])),10,(0,255,0),3)
        cv2.imwrite("img/{}_e{:02d}.jpg".format(imname,e),image)
    
    model.save("model_rnn.h5")

if __name__ == "__main__":
    main()    
