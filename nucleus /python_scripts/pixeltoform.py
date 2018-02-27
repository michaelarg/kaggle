import os,sys
import collections
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import re
from os import listdir
import glob
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
'''
20/02/18 MA
*variable names need to be redone as they are pretty silly/meaningless at the moment
This script is written for the kaggle competition of classifying nuclei.
They have a particular format for the submission scoring. Ex. 32 3 means
start at pixel 32 and then 33 and 34.
It reads in a given image and then converts it to this form.
To check that this is correct I will then convert the submission form back
into a matrix form to check that it is correct by printing the matrix.


ToDo: This seems to only be working with square matrices. 
'''


np.set_printoptions(threshold=np.nan)
def to_form(img):
    a = np.nonzero(img) #find all the indices that are non zero in the image
    #print a
    a = zip(a[1], a[0])
    a = sorted(a)
    #print a
    a = [list(i) for i in a]
    b = [i[0]*img.shape[0]+1+i[1] for i in a]
    
    count_list = []
    novals = []
    novals.append(b[0]) #the first value is put in
    
    novals.pop(0)

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
   # print "type of", type(kk)
    return kk

def from_formtest(subform, img):
    print "shape of img" , np.shape(img)[1]
    #print "subform", subform[0]
    print type(subform)
    subform = subform.split()
 #   print subform
    k = np.array(subform)
    k= k.astype(int)

    masks = k.reshape([len(subform)/2,2])
   # print masks
    master = []
    for i in range(len(masks)):
        sub = []
        for j in range(masks[i][1]):
            sub.append(masks[i][0] + j)
        master.append(sub)

    C = [item for sublist in master for item in sublist] #very efficient list operations!
    print "length of C", len(C)
    Z = np.repeat(1,len(C))
    er = zip(C,Z)

    print "length of er" , len(er)

    blank = [range(0,np.shape(img)[0] * np.shape(img)[1])] #big list! index
    blank = [item for sublist in blank for item in sublist]

    zilch = [np.repeat(0, np.shape(img)[0] * np.shape(img)[1] )] #zeros
    zilch = [item for sublist in zilch for item in sublist]
    
    

    dds = zip(blank, zilch )

    dds = [list(elem) for elem in dds]
    er = [list(elem) for elem in er] #tuples to lists
    pwq = [list(e) for e in zip([blank,zilch])] #big list of 256*256 values all zero now

    out_tup = [item for item in er if item[1] not in dds[0] ] #outtup is a list of each pixel from sub form and a 1

    countit=0
    for i in er:
        try:
            dds[i[0]-1][1] = 255
            countit += 1
        except IndexError:
            print "trying to access", i

    pixelvals = [index[1] for index in dds]
    print pixelvals[1000:2000]


    pixel_shape = np.reshape(pixelvals, [np.shape(img)[1],np.shape(img)[0]])
 #  test = pixel_shape.T
    test = pixel_shape
    return test

def from_formtest_nonsquare(subform, img):
    print "shape of img" , np.shape(img)[0]
    print "shape of img" , np.shape(img)[1]

    #print "total number of pixels" , np.shape(img)[0] * np.shape(img)[1]

    pix_count = np.shape(img)[0] * np.shape(img)[1]

    print "total number of pixels" , pix_count


    #print "subform", subform[0]
    print type(subform)

   # print "what is read in", subform

    
    #print subform
    subform = subform.split()
    #print subform
    k = np.array(subform)
    k= k.astype(int)
    #print np.shape(k)[0]
 #   print np.shape(k)

    masks = k.reshape([len(subform)/2,2])
   # print masks
   # print masks
    master = []
    for i in range(len(masks)):
        sub = []
        for j in range(masks[i][1]):
            sub.append(masks[i][0] + j)
        master.append(sub)

    master = sorted(master)
    print len(master)
    #Try to remove possible duplicates but that really should be a problem
 #   ab = list(set(master))    
#    print len(ab)
    
    #print "master",master[1]

    C = [item for sublist in master for item in sublist] #very efficient list operations!
    print "length of C", len(C)
    Z = np.repeat(1,len(C))
    er = zip(C,Z)

 #  print er[1:20]

 #  print "length of er" , len(er)

    blank = [range(0,pix_count)] #big list! index
    blank = [item for sublist in blank for item in sublist]

 #   print len(blank)

    zilch = [np.repeat(0, pix_count )] #zeros
    zilch = [item for sublist in zilch for item in sublist]

    print "length of zilch" , len(zilch)

    dds = zip(blank, zilch )
 #  print dds[0:10]
    dds = [list(elem) for elem in dds]
 #   print "dds" , dds[0:10]
    er = [list(elem) for elem in er] #tuples to lists
    pwq = [list(e) for e in zip([blank,zilch])] #big list of 256*256 values all zero now

    out_tup = [item for item in er if item[1] not in dds[0] ] #outtup is a list of each pixel from sub form and a 1


 #   print er[1:20]


    countit=0
    for i in er:
        try:
            #print i
            dds[i[0]-1][1] = 255  #dds is a list of lists each with an index from 0 to last pixel

            #print "this is i", i[0]-1            
    
            countit += 1
        except IndexError:
            print "trying to access", i

    print "DDS", dds[62785:62790]
    pixelvals = [index[1] for index in dds]

 #   print "pixelvals" , pixelvals[62785:62790]
    print "pixelvals", len(pixelvals)

    print "pplease be it!"
    print np.shape(img)[0]
    print np.shape(img)[1]
    pixel_shape = np.reshape(pixelvals, [np.shape(img)[1],np.shape(img)[0]])
    test = pixel_shape

 #   print test
#    test = pixel_shape

   # print "TEST", np.nonzero(test)


    return test.T

def plot_test(test, real):
    
    print "are plots equal", np.array_equal(test,real)
    print np.count_nonzero(test)
    print np.count_nonzero(real)

    print "test",   np.nonzero(test)[0][0]
    print "test",   np.nonzero(test)[1][0]

    print "real",   np.nonzero(real)[0][0]
    print "real",   np.nonzero(real)[1][0]

    #print "shape of test" , np.shape(test)[0]
    #print "shape of test" , np.shape(test)[1]

    #print "shape of real" , np.shape(real)[0]
    #print "shape of real" , np.shape(real)[1]

    #print "test",   np.nonzero(test)
   # print "real",   np.nonzero(real)


    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.gray()
    plt.imshow(test)
    f.add_subplot(1,2,2)
    plt.imshow(real)
    plt.title("actual")
    plt.gray()
    plt.show(block=True)


def mask_flatten(dir):
    image_paths = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.png')]
    pixelbank = []
    for img in image_paths:
        img = cv2.imread(img,0) #the image read in is a grayscale image
 #       plot_test(img,img)
        
        tmp = to_form(img)
        pixelbank.append(tmp)
 #       pixelbank = unlist(pixelbank)

    pixelbank = str(pixelbank)[1:-1]
    pixelbank = pixelbank.replace(',', '')
    pixelbank = pixelbank.replace("'", '')
    return pixelbank
        
    

def main():
    os.chdir("/Users/michaelargyrides/Documents/datascience_nuclei/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks")
    img_path = "/Users/michaelargyrides/Documents/datascience_nuclei/stage1_train/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe/masks"
    image_paths = [os.path.join(img_path, x) for x in os.listdir(img_path) if x.endswith('.png')]
    img = cv2.imread(image_paths[0],0)


#    print img_path
 #   print image_paths
    
    a = mask_flatten(img_path)

 #   print "a", a

    coolmat = from_formtest_nonsquare(a, img)


   # print np.array_equal(img,coolmat)


    plot_test(coolmat, img)


    os.chdir("..")
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(coolmat)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(img_path+'_total.png', bbox_inches=extent)
#    plt.savefig(img_path+"master.png", frameon=False)

if __name__ == '__main__':
    main()




















