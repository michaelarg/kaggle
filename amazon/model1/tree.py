import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import pandas as pd
import numpy as np
import pydotplus
import cv2
import keras
import scipy
import shutil
import glob
import tensorflow as tf
import scipy.stats.stats as st
import csv
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
#from sklearn.neural_network import MLPClassifier
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 


os.chdir('/Users/michaelargyrides/Desktop/Kaggle')
train = pd.read_csv("features.csv")

trainx = train.iloc[:,1:7]
trainy = train.iloc[:,7:24]

vals = list(train)
feats=vals[1:7]
cats = vals[7:24]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainx,trainy)

#result = clf.predict([[18,130,40,23,0.5,0.1]])

from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=feats,  
                         class_names=cats,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  
