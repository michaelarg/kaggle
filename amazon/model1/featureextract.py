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
import time
import re
os.chdir('/Users/michaelargyrides/kaggle')
ko = pd.read_csv("binmap.csv")

os.chdir("/Users/michaelargyrides/Desktop/Kaggle/train-jpg")
dir = "/Users/michaelargyrides/Desktop/Kaggle/train-jpg"

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

features=['var', 'zerocount', 'mean' , 'median', 'skew', 'kurtosis','iqr']
imagefiles=[]

for file in sorted(glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train-jpg/*.jpg"), key=numericalSort):
    x = file
    y = x.replace("/Users/michaelargyrides/Desktop/Kaggle/train-jpg/",'')
   # print y
    imagefiles.append(y)

fmat = pd.DataFrame(0, index= imagefiles, columns=features).astype(float)

ind=0
count=0
for file in sorted(glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train-jpg/*.jpg"), key=numericalSort):
    start_time = time.time()
    
 # print file

    img = cv2.imread(file)
 

    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

 
    h, s, v = cv2.split(hsv_img)

 
    hvals=h.flatten()
    
 #   start_time = time.time()
#    counts, bins, bars = plt.hist(hvals, 179)
   # print("--- %s colour conversion image ---" % (time.time() - start_time))

    #Keep this in for demonstration purposes.

    n,bins = np.histogram(hvals, range(180))    

#    print n   
 #   print bins
 #   print len(bins)
  
    count_it = n.tolist()    
    zeroval = count_it.count(0)
 #   print zeroval
    #zero bin counts
##    count = 0
##    for i in range(len(counts)):
##        if counts[i] == 0:
##            count=count+1
           
    #print count

    q75, q25 = np.percentile(hvals , [75,25])
    iqr = q75 - q25    
    
    fmat['var'][ind] = np.var(hvals)
    fmat['zerocount'][ind] = zeroval
    fmat['mean'][ind] = np.mean(hvals)
    fmat['median'][ind] = np.median(hvals)
    fmat['skew'][ind] = st.skew(hvals)
    fmat['kurtosis'][ind] = st.kurtosis(hvals)
    fmat['iqr'][ind] = iqr
    ind=ind+1    
    print("--- %s seconds ---" % (time.time() - start_time))

    print "end of this loop"

fmat.to_csv("trainFin.csv", encoding = 'utf-8')

