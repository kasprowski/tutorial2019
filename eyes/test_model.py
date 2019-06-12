import os
import cv2
from sklearn.metrics.regression import mean_absolute_error
from tensorflow.python.keras.models import load_model
import numpy as np

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


def load_images(indir):
    samples = []
    labels = []
    for imagePath in os.listdir(indir):
        print(imagePath)
        image = cv2.imread("{}/{}".format(indir,imagePath))
        image = preprocess(image)
        samples.append(image)
        label = np.zeros((2))
        label[0] = float(imagePath[0:imagePath.find("_")])
        label[1] = float(imagePath[imagePath.find("_")+1:imagePath.find(".")])
        print(label[0],label[1])
        labels.append(label)
        
    samples = np.array(samples, dtype="float")
    labels = np.array(labels)
    return samples,labels



testSamples,testLabels = load_images("eye_left")

model = load_model("model.01-45.01.h5") ### add the correct model name!!!

predictions = model.predict(testSamples)

results = np.zeros((800,1500,3))

# create an image with current predictions
for i in range(testSamples.shape[0]):
    cv2.circle(results,(int(testLabels[i,0]),int(testLabels[i,1])),10,(0,255,0),2)
    cv2.circle(results,(int(predictions[i,0]),int(predictions[i,1])),10,(255,0,0),2)
    cv2.line(results, 
             (int(predictions[i,0]),int(predictions[i,1])),
             (int(testLabels[i,0]),int(testLabels[i,1])), 
             (255,0,0),2
             )
cv2.imwrite("test_model.jpg",results)
 
print("Final MAE: {}".format(mean_absolute_error(testLabels,predictions)))    
input("")