'''
Prepares images from eye trcking data files

@author: pawel@kasprowski.pl
'''
import numpy as np
import os
import cv2

indir = "radio"
outdir = "radio_img"
samplesIMG = []
namesIMG = []
i=0
added = 0
for file in os.listdir(indir):
    sample = np.genfromtxt(os.path.join(indir, file), delimiter=',', skip_header=1)
    print("{}. {} - {}".format(i,file,sample.shape))
    if sample.shape[0]>100:
        samplesIMG.append(sample)
        namesIMG.append(file)
        added+=1
    i+=1
print("Added {} of {}".format(added,i))    

for j in range(len(samplesIMG)):
    print(namesIMG[j])
    img = np.zeros((108,192))
    s = samplesIMG[j]
    for i in range(s.shape[0]):
        x = int(s[i,3]/10)
        y = int(s[i,4]/10)
        #print("{} {},{}".format(s[i,0],x,y))
        if x>=0 and y>=0 and x<192 and y<108:
            img[y,x] = 255
            
    img = cv2.GaussianBlur(img,(9,9),5)        
    cv2.imwrite(outdir+'/'+namesIMG[j]+".jpg", img)
        