'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Module using models loaded from models.py to classify samples
Uses pyplot to show loss and metrics in subsequent training epochs

@author: pawel@kasprowski.pl
'''

import models
import matplotlib.pyplot as plt
import numpy as np

def classification(model_name,samples,labels,rangex,rangey):
    samples = np.array(samples)
    labels = np.array(labels)

    #available models from cl_models.py    
    nnmodels = {
        "PERCEPTRON": models.perceptron(),
        "HIDDEN": models.hidden_linear(),
        "HIDDEN_RELU": models.hidden_sigmoid()
    }
    model = nnmodels.get(model_name)
    
    model.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy'])
    
    # fit model
    EPOCHS = 100
    H = model.fit(samples, labels, epochs=EPOCHS,batch_size=10)    

    print("classifier ",model," - created")

    # classify each cell in area
    result = np.zeros([rangex,rangey])
    for x in range(rangex):
        for y in range(rangey):
            sample = np.array([x,y])
            r = model.predict(sample.reshape(1, -1))
            result[x][y]=r
#             if r>0.5:
#                 result[x][y]=1
#             else:
#                 result[x][y]=0
                 

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    return result            
        

