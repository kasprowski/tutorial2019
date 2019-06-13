'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Checks the model saved by 'animals_rnn.py'
Saves genuine (*_real.jpg) and generated (*_gen.jpg) scanpaths in /scanpaths folder

@author: pawel@kasprowski.pl
'''
import cv2
import os
import numpy as np
from tensorflow.python.keras.models import load_model

def load_files(indir):
    samples = []
    names = []
    for file in os.listdir(indir):
        sample = np.genfromtxt(os.path.join(indir, file), delimiter=',')
        samples.append(sample)
        names.append(file[:-4])
    return samples,names

# creates an image with scanpaths taken from a sample
# and saves it in /scanpaths folder
def save_scanpath(sample,prefix):
    imname = "animal-11"    
    image = cv2.imread("img/{}.jpg".format(imname))
    ox=0
    oy=0
    for j in range(sample.shape[0]):
        sx = int(sample[j,1])
        sy = int(sample[j,2])
        if sx<=0 or sy<=0: continue
        if j>0: cv2.line(image,(ox,oy),(sx,sy),(255,0,0),3)
        cv2.circle(image,(sx,sy),10,(0,0,255),3)
        ox = sx
        oy = sy
    cv2.imwrite("scanpaths/{}_{}_sp.jpg".format(prefix,imname),image)
    

def main():
    model = load_model("model_rnn.h5")
    samples,names = load_files("data")
    
    for i in range(len(samples)):
        sample = samples[i]
        name = names[i]
        print("Processing",name)
        #genuine scanpath
        save_scanpath(sample, "{}_real".format(name))
        sampleSequence = sample[0:20,1:]
        sampleSequence = np.expand_dims(sampleSequence, axis=0)
        ts = 0
        sequence = []
        # predict subsequent samples based on 20 samples from genuine sample
        for i in range(sample.shape[0]-20):
            predictions = model.predict(sampleSequence)
            x = np.roll(sampleSequence,-1,axis=1)
            x[0,19] = sample[i+20,1:]
            sampleSequence = x
              
            item = np.zeros((3))
            item[0] = ts
            item[1] = predictions[0,0]
            item[2] = predictions[0,1] 
            sequence.append(item)
            ts += 1
        sequence = np.array(sequence)
        save_scanpath(sequence, "{}_gen".format(name))

if __name__ == "__main__":
    main()    
