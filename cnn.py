import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import numpy as np

import keras
import scipy
import shutil
import pickle
import tensorflow as tf
import time
import re
from scipy import misc
import sklearn
from scipy import ndimage
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
#from keras.datasets import mnist
import pprint
import glob
from keras.optimizers import SGD
from sklearn.metrics import fbeta_score
from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath="weights.best.hdf5"

os.chdir('/home/tester/kerasnn/')

trainy=pd.DataFrame.from_csv("trainM.csv")
print trainy.shape
trainy.ix[:,1:17]
trainy=np.array(trainy)

loaded = np.load('64img.npz')
trainx = loaded['trainx']
testx = loaded['testx']
test2x = loaded['test2x']

trainx, valx, trainy, valy = train_test_split(trainx, trainy, test_size=0.1, random_state=42)


callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)]

model = Sequential()
model.add(BatchNormalization(input_shape=(64, 64,3)))
model.add(Conv2D(32 ,(3, 3), activation='relu',padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same' , activation = 'relu'))
model.add(Conv2D(64, (3,3), activation  = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding = 'same' , activation = 'relu'))
model.add(Conv2D(128, (3,3), activation  = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding = 'same' , activation = 'relu'))
model.add(Conv2D(256, (3,3), activation  = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(17))
model.add(Activation('sigmoid'))

 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)]


start=time.time()
model.fit(trainx, trainy,batch_size=128, epochs=50, verbose=1,callbacks=callbacks, shuffle=True,  validation_data=(valx, valy))
end=time.time()-start
print "took this long",end

#So batch_size = 128, means that 128 samples are training in one pass of the model, then the next 128 and after all are finished we go to the next epoch
validresult = model.predict(valx, batch_size = 128, verbose=2)

print validresult

print(fbeta_score(valy, np.array(validresult)>0.2, beta=2, average='samples'))

pred = model.predict(testx.reshape((len(testx), 64,64,3)))
predictions=np.round(pred)

pred2 = model.predict(test2x.reshape((len(test2x), 64, 64,3)))
predictions2=np.round(pred2)

predictions = np.vstack((predictions,predictions2))
np.savetxt("predictions64.csv",predictions, delimiter=',',fmt='%.14f')

