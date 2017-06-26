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
