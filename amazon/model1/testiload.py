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
destination = '/Users/michaelargyrides/Desktop/Kaggle/train_set'

img = cv2.imread('/Users/michaelargyrides/Desktop/Kaggle/train_set/train_0.jpg')
print img
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
