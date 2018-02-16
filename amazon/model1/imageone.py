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


os.chdir('~/images/test-jpg')

img = (cv2.imread('~/images/test-jpg'))
hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img) 
hvals=h.flatten()
b, bins, patches = plt.hist(hvals, 179)


