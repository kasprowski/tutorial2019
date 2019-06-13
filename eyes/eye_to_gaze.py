'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Uses images from /eye_left to find gaze coordinates

@author: pawel@kasprowski.pl
'''

import os
import cv2
import numpy as np

from sklearn.metrics.regression import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from tensorflow.keras.models import Sequential

# loads all images from /indir
# label is derived from file name
def load_images(indir):
    samples = []
    labels = []
    for imagePath in os.listdir(indir):
        image = cv2.imread("{}/{}".format(indir,imagePath))
        image = preprocess(image)
        samples.append(image)
        label = np.zeros((2))
        label[0] = float(imagePath[0:imagePath.find("_")])
        label[1] = float(imagePath[imagePath.find("_")+1:imagePath.find(".")])
        labels.append(label)
        
    samples = np.array(samples, dtype="float")
    labels = np.array(labels)
    return samples,labels

# resizes each image to (64,64) and then masks the image with ellipse
def preprocess(image):
    image = cv2.resize(image, (64, 64))
    mask = np.zeros_like(image)
    rows, cols,_ = mask.shape
    mask=cv2.ellipse(mask, center=(rows//2, cols//2), axes=(28,14), 
                     angle=0, startAngle=0, endAngle=360, 
                     color=(255,255,255), thickness=-1)
    result = np.bitwise_and(image,mask)
    result = result[14:64-14,:]
    return result

def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding="same",input_shape=(36,64,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    #model.add(Dropout(0.5))
    model.add(Dense(2)) # two values X,Y
    return model



print("Loading images...")
samples,labels = load_images("eye_left")
    
(trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.15, random_state=42)

model = build_cnn_model()
model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mae"])
checkpt1 = ModelCheckpoint(filepath='models/model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True)

# train
EPOCHS = 100
for e in range(EPOCHS):
    print("=" * 50)
    print("Iteration: {}".format(e))
    H = model.fit(trainSamples, trainLabels, validation_data=(testSamples, testLabels), epochs=1, batch_size=32, verbose=0,
                  callbacks=[checkpt1])
    print("Error so far: {}".format(H.history["val_mean_absolute_error"]))

    predictions = model.predict(testSamples, batch_size=32)
    results = np.zeros((800,1500,3))
    dx = 0
    dy = 0
    # create an image with current predictions
    for i in range(testSamples.shape[0]):
        cv2.circle(results,(int(testLabels[i,0]),int(testLabels[i,1])),10,(0,255,0),3) # true gaze points
        cv2.circle(results,(int(predictions[i,0]),int(predictions[i,1])),10,(255,0,0),3) # predicted gaze point
        cv2.line(results, 
                 (int(predictions[i,0]),int(predictions[i,1])), # line connecting ground truth with prediction
                 (int(testLabels[i,0]),int(testLabels[i,1])), 
                 (255,0,0),3
                 )
        dx = dx + abs(testLabels[i,0] - predictions[i,0])
        dy = dy + abs(testLabels[i,1] - predictions[i,1])
        
    dx = dx/testSamples.shape[0]
    dy = dy/testSamples.shape[0]
        
    cv2.putText(results,"ErrX: {0:.0f}".format(dx),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)    
    cv2.putText(results,"ErrY: {0:.0f}".format(dy),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)    
    cv2.imwrite("img/test_{:02d}.jpg".format(e),results)
 
print("Final MAE: {}".format(mean_absolute_error(testLabels,predictions)))    

model.save("final_model_{0:.2f}".format(mean_absolute_error(testLabels,predictions)))
input("")


