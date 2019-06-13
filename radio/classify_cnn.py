'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Loads images using loader.py and performs CNN classification

@author: pawel@kasprowski.pl
'''
import numpy as np
import loader

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.classification import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.utils import class_weight
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

# creates and compiles the CNN model
def cnn_model(inputShape,numClasses):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("sigmoid"))
    model.add(Dense(256))
    model.add(Activation("sigmoid"))
    model.add(Dense(numClasses))
    model.add(Activation("softmax"))

    loss='categorical_crossentropy'    
    model.compile(loss=loss, optimizer="adam",metrics=['accuracy'])
    return model


def main():
    samples,labels,_ = loader.load_img("radio_img")

    #add the fourth dimension (color)
    samples = np.expand_dims(samples, axis=4)

    print("shape = {}".format(samples.shape))
    inputShape = (samples.shape[1],samples.shape[2],samples.shape[3])
    print("inputShape = {}".format(inputShape))

    #weights
    class_weights = class_weight.compute_class_weight('balanced',np.unique(labels),labels)
    d_class_weights = dict(enumerate(class_weights))
    print("weights {}".format(d_class_weights))

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    classesNum= labels.shape[1]
    print ("Classes: {}".format(classesNum))

    #split to training and test
    (trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, test_size=0.25, random_state=42)
    
    model = cnn_model(inputShape,classesNum)

## checkpoints
#    checkpt1 = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True)
#    checkpt2 = EarlyStopping(monitor='val_loss', patience=3)


    EPOCHS=20
    BATCH=50
    model.fit(trainSamples, trainLabels, batch_size=BATCH, epochs=EPOCHS,class_weight=d_class_weights,verbose=1,
              #callbacks = [checkpt1,checkpt2],
              validation_data=(testSamples,testLabels))
    
    cnnResults = model.predict(testSamples)

    print(confusion_matrix(testLabels.argmax(axis=1), cnnResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), cnnResults.argmax(axis=1)))
    cnnAcc = accuracy_score(testLabels.argmax(axis=1), cnnResults.argmax(axis=1))
    print("Accuracy CNN: {:.2f}".format(cnnAcc))
    print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(testLabels.argmax(axis=1), cnnResults.argmax(axis=1))))
    input("")

if __name__ == "__main__":
    main()