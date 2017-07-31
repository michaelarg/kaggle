import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import numpy as np
import cv2
import keras
import scipy
import shutil
import pprint, pickle
#from keras.datasets import mnist
import pprint
import hickle as hkl
import glob
import h5py
from keras.optimizers import SGD
from sklearn.metrics import fbeta_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
import re
#os.chdir('/Users/michaelargyrides/Desktop/Kaggle/kerasnn')

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



trainimg = sorted(glob.glob('/Users/michaelargyrides/Desktop/Kaggle/kerasnn/train-jpgM/*.jpg'), key=numericalSort)
trainx = np.array([np.array( cv2.resize(cv2.imread(file), (28, 28))) for file in trainimg])

trainx = trainx.astype('float32')
trainx = trainx/255

testimg = sorted(glob.glob('/Users/michaelargyrides/Desktop/Kaggle/kerasnn/test-jpgM/*.jpg'), key=numericalSort)
testimg2 = sorted(glob.glob('/Users/michaelargyrides/Desktop/Kaggle/kerasnn/test-jpg-addtionalM/*.jpg'), key=numericalSort)

testx = np.array([np.array( cv2.resize(cv2.imread(file), (28, 28))) for file in testimg])
testx = testx.astype('float32')
testx = testx/255

test2x = np.array([np.array( cv2.resize(cv2.imread(file), 28, 28)) for file in testimg2])
test2x = test2x.astype('float32')
test2x = test2x/255

np.savez_compressed('28img.npz',trainx=trainx,testx=testx,test2x=test2x)

#loaded = np.load('32img.npz')
#trainx = loaded['trainx']
#testx = loaded['testx']
#test2x = loaded['testx']

##pkl_file = open('trainx_28.pkl', 'rb')
##
##data1 = pickle.load(pkl_file)
##pprint.pprint(data1)
##
##pkl_file.close()
##
##h5f = h5py.File('data.h5', 'w')
##h5f.create_dataset('dataset_1', data=data1)
##h5f.close()
##
##hkl.dump(data1, 'new_data_file.hkl')
##
##np.save('test.npy', data1)
