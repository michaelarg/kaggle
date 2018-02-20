import os,sys
import collections
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import re
from os import listdir
import glob
import matplotlib.pyplot as plt

'''
20/02/18 MA
*variable names need to be redone as they are pretty silly/meaningless at the moment
This script is written for the kaggle competition of classifying nuclei.
They have a particular format for the submission scoring. Ex. 32 3 means
start at pixel 32 and then 33 and 34.
It reads in a given image and then converts it to this form.
To check that this is correct I will then convert the submission form back
into a matrix form to check that it is correct by printing the matrix.
'''

np.set_printoptions(threshold=np.nan)
def to_form(img):
    a = np.nonzero(img) #find all the indices that are non zero in the image

    a = zip(a[1], a[0])
    a = sorted(a)
    a = [list(i) for i in a]
    b = [i[0]*img.shape[0]+1+i[1] for i in a]
    
    count_list = []
    novals = []
    novals.append(b[0]) #the first value is put in
    
    novals.pop(0)
    
  #  print novals
 #   print 'length of list' , len(b)
    count = 1
    for i in range(len(b)-1):
        if b[i] + 1 == b[i+1]:
            if count == 1:
                novals.append(b[i])
            count +=1
        else:
            count_list.append(count)
            count = 1

    if (b[len(b)-1]-1) != b[len(b)-2]:
        print "they are not consecutive bra"
        novals.append(b[len(b)-1])
        count_list.append(1)
          
    count_list.append(count)

    kk = zip(novals, count_list)

    kk = re.sub(r'[^\w]', ' ', str(kk))      
    kk = " ".join(kk.split())

    return kk


def from_formtest(subform, img):
    subform = subform.split()
 #   print subform
    k = np.array(subform)
    k= k.astype(int)

    masks = k.reshape([len(subform)/2,2])

    master = []
    for i in range(len(masks)):
        sub = []
        for j in range(masks[i][1]):
            sub.append(masks[i][0] + j)
        master.append(sub)

    C = [item for sublist in master for item in sublist] #very efficient list operations!
    Z = np.repeat(1,len(C))
    er = zip(C,Z)

    blank = [range(0,np.shape(img)[0] * np.shape(img)[0])] #big list! index
    blank = [item for sublist in blank for item in sublist]

    zilch = [np.repeat(0, np.shape(img)[0] * np.shape(img)[0] )] #zeros
    zilch = [item for sublist in zilch for item in sublist]

    dds = zip(blank, zilch )

    dds = [list(elem) for elem in dds]
    er = [list(elem) for elem in er] #tuples to lists
    pwq = [list(e) for e in zip([blank,zilch])] #big list of 256*256 values all zero now

    out_tup = [item for item in er if item[1] not in dds[0] ] #outtup is a list of each pixel from sub form and a 1

    countit=0
    for i in er:    #print i[0]
       # print i[0]-1
      
        dds[i[0]-1][1] = 255
        countit += 1

    pixelvals = [index[1] for index in dds]

    pixel_shape = np.reshape(pixelvals, [np.shape(img)[0],np.shape(img)[0]])
    test = pixel_shape.T
    return test

def plot_test(test, real):
    print "are plots equal", np.array_equal(test,real)
    print np.count_nonzero(test)
    print np.count_nonzero(real)

    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.gray()
    plt.imshow(test)
    f.add_subplot(1,2,2)
    plt.imshow(real)
    plt.title("actual")
    plt.gray()
    plt.show(block=True)


#print np.array_equal(img, test)

def main():
    os.chdir("/Users/michaelargyrides/Documents/datascience_nuclei/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks")
    img = cv2.imread('f61af9040d085e725fafe93e102ee343bf8528651e8ad2cbc23487984db16e01.png',0) #the image read in is a grayscale image

    subform = to_form(img)
    print subform
    coolmat = from_formtest(subform, img)

    plot_test(coolmat, img)


if __name__ == '__main__':
    main()




















