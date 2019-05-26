'''
Uses data from images and from a flat csv file to classify samples

@author: pawel@ksasprowski.pl
'''
import loader

from sklearn.metrics.classification import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing.label import LabelBinarizer
from tensorflow.python.keras.models import Model

import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential

# load data
samplesIMG,labels,namesIMG = loader.load_img("radio_img")
samplesCSV,labelsCSV,namesCSV = loader.load_csv("radio.csv")

# find pairs for samplesIMG in samplesCSV
samples_paired = []
for i in range(samplesIMG.shape[0]):
    for j in range(samplesCSV.shape[0]):
        if namesCSV[j]==namesIMG[i]:
            samples_paired.append(samplesCSV[j])
            
samplesCSV = np.array(samples_paired)
samplesIMG = np.expand_dims(samplesIMG, axis=3)

print("Paired")
print("Samples IMG: {}".format(len(samplesIMG)))
print("Samples CSV: {}".format(len(samplesCSV)))

# one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
numClasses = labels.shape[1]


inputShape = (108,192,1) #samplesIMG.shape

#model for images
cnnmodel = Sequential()
cnnmodel.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape))
cnnmodel.add(Activation("relu"))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Conv2D(32, (3, 3), padding="same"))
cnnmodel.add(Activation("relu"))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Dropout(0.25))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(16))
cnnmodel.add(Activation("relu"))

# model for features (CSV)
flatmodel = Sequential()
flatmodel.add(Flatten(input_shape=(14,)))
flatmodel.add(Dense(50, activation='sigmoid'))

# concatenated model
combined = concatenate([cnnmodel.output, flatmodel.output])
x = Dense(16, activation="sigmoid")(combined)
x = Dense(numClasses, activation="sigmoid")(x)

model = Model(inputs=[cnnmodel.input, flatmodel.input], outputs=x)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

EPOCHS = 20
BATCH=100
model.fit([samplesIMG,samplesCSV], labels, batch_size=BATCH, epochs=EPOCHS,validation_split=0.25)

results = model.predict([samplesIMG,samplesCSV])

print(confusion_matrix(labels.argmax(axis=1), results.argmax(axis=1)))
print(classification_report(labels.argmax(axis=1), results.argmax(axis=1)))
print("Accuracy: {}".format(accuracy_score(labels.argmax(axis=1), results.argmax(axis=1))))

