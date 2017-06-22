import os,sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import numpy as np
import cv2
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

mat=pd.DataFrame.from_csv("train_v2.csv")
mat.index.name = 'image'
mat.reset_index(inplace=True)
mat['tags'].values.tolist()

columns=['agriculture','clear','cloudy','primary','road','shifting','water','partly_cloudy','haze','habitation','slash_burn','cultivation','blooming',
         "bare_ground",'selective_logging','conventional_mine','artisinal_mine','blow_down']

hmat = pd.DataFrame(0, index=mat['image'], columns=columns)

for index, row in mat.iterrows():
    liststr = str.split(row['tags']," ")
    for tag in liststr:
        hmat[tag].ix[index]=1

hmats = hmat.sum(1) 
countmat = hmats.value_counts(normalize=False)
countmat_vals = countmat.index.values.tolist()
fig , ax1 = plt.subplots(1,1)
plt.bar(range(len(countmat)), countmat, align='center', alpha=0.5)
plt.title("tag count frequency")
plt.xticks(range(len(countmat_vals)),countmat_vals)
plt.show()


heat = hmat.groupby(columns).size().reset_index(name="frequency")
heat = heat.sort('frequency',axis=0,ascending=False)
heatplot = heat[0:20]
heatplot = heatplot.iloc[:,:-1]


fig , ax1 = plt.subplots(1,1)
ax1.imshow(heatplot,interpolation='nearest')
ax1.set_xticks(range(18))
ax1.set_yticks( range(20) )
ax1.set_xlabel('Tags')
ax1.set_ylabel('Image Count')
ax1.set_yticklabels(heat.frequency[0:20], fontsize=10)
ax1.set_xticklabels(columns, fontsize=10, rotation='vertical')
plt.show()

#Clear, partly cloudy, cloudy and haze
# Each chip will have one and potentially more than one atmospheric label
# and zero or more common and rare labels
# cloudy should have no other labels possible errors
