import os,sys
import collections
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import re

#os.chdir("/Users/michaelargyrides/Documents/datascience_nuclei/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/images")
#img = cv2.imread('0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png',0) #the image read in is a grayscale image

os.chdir("/Users/michaelargyrides/Documents/datascience_nuclei/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks")
img = cv2.imread('0adbf56cd182f784ca681396edc8b847b888b34762d48168c7812c79d145aa07.png',0) #the image read in is a grayscale image

a = np.nonzero(img)
b = zip(a[1]*256+1,a[0])
c = [i[0] + i[1] for i in b]
c = sorted(c)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


def conseq_count(list):
    count_list = []
    novals = []
    novals.append(list[0])
    count = 1
    for i in range(len(list)-1):
        if list[i] + 1 == list[i+1]:
            if count == 1:
                novals.append(list[i])
            count +=1
        else:
            count_list.append(count)
            count = 1
    count_list.append(count)
    print len(novals)
    return zip(novals, count_list)


dd = conseq_count(c)

dd = re.sub(r'[^\w]', ' ', str(dd))      
dd = " ".join(dd.split())

#print dd
dd = dd.split()
k = np.array(dd)
k= k.astype(int)
masks = k.reshape([22,2])


master = []
for i in range(len(masks)):
    sub = []
    for j in range(masks[i][1]):
        sub.append(masks[i][0] + j)
    master.append(sub)


C = [item for sublist in master for item in sublist] #very efficient list operations!
Z = np.repeat(1,len(C))

er = zip(C,Z)


blank = [range(0,np.shape(img)[0] * np.shape(img)[0]         )]
blank = [item for sublist in blank for item in sublist]

zilch =      [np.repeat(0, np.shape(img)[0] * np.shape(img)[0] )]
zilch = [item for sublist in zilch for item in sublist]

dds = zip(blank, zilch )

dds = [list(elem) for elem in dds]
er = [list(elem) for elem in er]

pwq = [list(e) for e in zip([blank,zilch])]

out_tup = [item for item in er if item[1] not in dds[0] ]

for i in er:    #print i[0]
    dds[i[0]][1] = 1

pixelvals = [index[1] for index in dds]

pixel_shape = np.reshape(pixelvals, [np.shape(img)[0],np.shape(img)[0]])

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#rew = [dd

#    dds = del dds(i)
 #   dds.append(er[i])
    
    
#    print er[i] in dds
   # if er[i] in dds == True:
        
        #delete current value 'no' and replace with 'yes'
        
#if any(er[0][0] == dds[i][0] for i in range(0,len(dds))):
#    print "ya"




#er is the pixels
#dds is the blank pixels


'''
flatten a black image and overwrite the pixels that correspond to the values in C


'''




#pixel_mask = 

#indexpixel = np.array(dd)

#print b

#print a[1]

#counter=collections.Counter(a[1])
#print counter

#print (np.nonzero(img)[1]) #np.nonzero returns a tuple indicating the x,y coords of non zero pixels in the image.

