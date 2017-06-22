import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import numpy as np
import cv2
import glob
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir('/Users/michaelargyrides/kaggle/')

source = '/Users/michaelargyrides/kaggle/train-jpg'
dest1 = '/Users/michaelargyrides/kaggle/train_set'

ko = pd.read_csv("binmap.csv")
ko['sum']= ko.sum(1)
kosub = ko.query('sum == 1')
sam=kosub.sample(20)
filenamevals = sam['image'].tolist()

for file in filenamevals:
    file =  file+'.jpg'
    print file

##os.chdir('/Users/michaelargyrides/Desktop/Kaggle/train_set')
##
##
##count=0
##files = [f for f in os.listdir('.') if os.path.isfile(f)]
##for f in files:
##    count = count +1
##    img = cv2.imread(f)
##    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
##    h, s, v = cv2.split(hsv_img) 
##    hvals=h.flatten()
##    b, bins, patches = plt.hist(hvals, 179)
##    plt.subplot(5,4,count)
##    plt.title(f)
##plt.show()
##
##count=0
##files = [f for f in os.listdir('.') if os.path.isfile(f)]
##for f in files:
##    count = count +1
##    img = cv2.imread(f)
##    plt.subplot(4,4,count)
##    plt.imshow(img)
##plt.show()
##
##
##
##
##
##
##
###images are always displayed as BGR.
#HSV range in opencv is [0,180] [0,255] [0,255] for h s v respectively

