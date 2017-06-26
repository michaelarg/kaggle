import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import numpy as np
import cv2
import shutil
import glob
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir('/Users/michaelargyrides/kaggle')
ko = pd.read_csv("binmap.csv")

os.chdir('/Users/michaelargyrides/Desktop/Kaggle/train-jpg')
destination = '/Users/michaelargyrides/Desktop/Kaggle/train_set/'

def quote(s1):
    return "'%s'" % s1

ko['sum']= ko.sum(1)
kosub = ko.query('sum == 1 and cloudy == 1' ) #This allows you to check which tags are unique

if len(kosub) > 25:
    sam=kosub.sample(26)
else:
    sam=kobsub.sample(len(kosub))

filenamevals = sam['image'].tolist()

for file in filenamevals:
    file =  file+'.jpg'
    #copy instead of move
    shutil.copy(file, destination)

os.chdir('/Users/michaelargyrides/Desktop/Kaggle/train_set')

##images = [cv2.imread(file) for file in glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train_set/*.jpg")]
##imagesfile = []
##[imagesfile.append(file) for file in glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train_set/*.jpg")]


images=[]
imagesfile=[]

for file in glob.glob("/Users/michaelargyrides/Desktop/Kaggle/train_set/*.jpg"):
    #print file
    images.append(cv2.imread(file))
    x = file
    y = x.replace("/Users/michaelargyrides/Desktop/Kaggle/train_set/",'')    
    imagesfile.append(y)
   # print y
    os.remove(file)

for i in range(25):
 #   print i
    hsv_img = cv2.cvtColor(images[i],cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img) 
    hvals=h.flatten()
    #print imagesfile[i]
    plt.subplot(5,5,i+1)
    plt.hist(hvals, 179)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.title(i, size = 6)
    plt.hold(True)
plt.show()









##count=0
##files = [f for f in os.listdir('.') if os.path.isfile(f)]
##for f in files:
##    #print len(files)
##    col=quote(f)
##    #print col
##    
##    djd = destination+f
##    print quote(djd)
## #   count = count +1
##    img = cv2.imread(djd)
##    print img
##   # print img
###    print img
##    cv2.imshow('image',img)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
##    h, s, v = cv2.split(hsv_img) 
##    hvals=h.flatten()
##    b, bins, patches = plt.hist(hvals, 179)
##    plt.subplot(5,4,count)
##    plt.title(f)
##plt.show()
#####images are always displayed as BGR.
###HSV range in opencv is [0,180] [0,255] [0,255] for h s v respectively
##


