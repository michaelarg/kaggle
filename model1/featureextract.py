import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import numpy as np
import cv2
import shutil
import glob
import scipy.stats.stats as st
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir('/Users/michaelargyrides/kaggle')
ko = pd.read_csv("binmap.csv")

os.chdir("/Users/michaelargyrides/Desktop/Kaggle/train_set")
dir = "/Users/michaelargyrides/Desktop/Kaggle/train_set"

onlyfiles = next(os.walk(dir))[2] #dir is your directory path as string
numoffiles =  len(onlyfiles)-1

feats = np.zeros(shape=(numoffiles,7))

features=['var', 'zerocount', 'mean' , 'median', 'skew', 'kurtosis']

imagefiles=[]

for file in glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train_set/*.jpg"):
    x = file
    y = x.replace("/Users/michaelargyrides/Desktop/Kaggle/train_set/",'')    
    imagefiles.append(y)

fmat = pd.DataFrame(0, index= imagefiles, columns=features).astype(float)


ind=0
for file in glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train_set/*.jpg"):
    #print file
    img = cv2.imread(file)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img) 
    hvals=h.flatten()
    counts, bins, bars = plt.hist(hvals, 179)

    #zero bin counts
    count = 0
    for i in range(len(counts)):
        if counts[i] == 0:
            count=count+1
    print count
    fmat['var'][ind] = np.var(hvals)
    fmat['zerocount'][ind] = count
    fmat['mean'][ind] = np.mean(hvals)
    fmat['median'][ind] = np.median(hvals)
    fmat['skew'][ind] = st.skew(hvals)
    fmat['kurtosis'][ind] = st.kurtosis(hvals)


##    x = file
##    y = x.replace("/Users/michaelargyrides/Desktop/Kaggle/train_set/",'')    
##    imagefiles.append(y)
    ind=ind+1    


