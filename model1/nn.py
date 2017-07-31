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
#from sklearn.neural_network import MLPClassifier
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
import numpy
os.chdir('/Users/michaelargyrides/Desktop/Kaggle')
train = pd.read_csv("features.csv")

test = pd.read_csv("featurestest.csv")
testx = test.iloc[:,1:7]
testx = testx.as_matrix()


#We normalise our data before we put it into the neural net so that it
#has a decent chance of converging

#Split Inputs and Outputs
trainx = train.iloc[:,1:7]
trainy = train.iloc[:,7:24]

trainx = trainx.as_matrix()
trainy = trainy.as_matrix()

print type(trainx)

model = Sequential() #Create the model - sequential is used for a linear stack of layers
model.add(Dense(12, input_dim=6, activation='relu')) #Give the model information about its shape
model.add(Dense(6, activation='relu'))
model.add(Dense(17, activation='sigmoid'))

#Compile is where we confgure the learning process for our model.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model.fit(trainx, trainy, epochs=1, batch_size=10)
print("--- %s seconds ---" % (time.time() - start_time))


#predictions=model.predict(testx)
#rounded = [round(x[0]) for x in predictions]
#print(rounded)


#scores.model.evaute(trainx,trainy)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#plot_model(model, show_shapes = True,  to_file='model.png')
