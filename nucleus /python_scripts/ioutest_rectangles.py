import os,sys
from pixeltoform import plot_test
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import matplotlib.pyplot as plt

'''
23/2/18 - MA
Description: here we write some to test the intersection over union metric that we need for object
detection or object segemntation.

How it works - calculates the non zero pixels in the image which is not going to work with real image segemntation

Input: two images - when actually using the algorithm one will be the train image with a box around the
area of interest. The other will be the predicted box.

Output: iou metric as a float and possibly the image of the two boxes drawn.

*Issues - right now the problem is that it is a solid box and the whole algorithm relies on it being
a solid box. What happens when the box is not filled in and there is an image in there? Probably fails.


*Todo - one remedy is to have the algoritm
'''


def box_shape(box):
    dims = np.nonzero(box)
    x1 = np.amin(dims[0])
    x2 = np.amax(dims[0])
    y1 =  np.amin(dims[1])
    y2 =  np.amax(dims[1])

    shapex = x2-x1
    shapey = y2-y1
    print "shapes x and y" , shapex,shapey
    return(x1,x2,y1,y2,shapex,shapey)
    
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_box(box1, box2, iimg, iimg2):
    img = np.zeros([150,150])
    
    
    x1,x2,y1,y2,shapex,shapey = box_shape(box1)
    print x1
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255), thickness = -1) #top left and bottom right

    x1,x2,y1,y2,shapex,shapey = box_shape(box2)
    print x2
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255), thickness = -1) #top left and bottom right

    ix1,ix2,iy1,iy2,ishapex,ishapey = overlap(iimg,iimg2)
    
    cv2.rectangle(img,(ix1,iy1),(ix2,iy2),(0,0,0), thickness = -1) #top left and bottom right 
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlap(img,img2):
    img = zip(np.nonzero(img)[0],np.nonzero(img)[1])
    img2 = zip(np.nonzero(img2)[0],np.nonzero(img2)[1])

    listless = [list(i) for i in img]
    listless2 = [list(i) for i in img2]

    intersection = [item for item in listless if item in listless2]

    xax = [item[0] for item in intersection]
    yax = [item[1] for item in intersection]

    ix1 = np.amin(xax)
    ix2 = np.amax(xax)
    iy1 =  np.amin(yax)
    iy2 =  np.amax(yax)
    shapex = ix2-ix1
    shapey = iy2-iy1
    
    return(ix1,ix2,iy1,iy2,shapex,shapey)

def iou_metric(box1 , box2, overlap):
    print box1
    print box2
    print overlap

    numerator=overlap[4]*overlap[5]
    print numerator
    denominator = (box1[4] * box1[5] + box2[4]* box2[5])-numerator
    print denominator
    return float(numerator)/float(denominator)

def main():
    os.chdir("/Users/michaelargyrides/Desktop/iou_test")
    img = cv2.imread('test1.png',0) #the image read in is a grayscale image
    img2= cv2.imread('test2.png',0)

    x1,x2,y1,y2,shapex,shapey = overlap(img,img2)

    draw_box(img,img2, img,img2)

    print iou_metric(box_shape(img),box_shape(img2),overlap(img,img2))

if __name__ == '__main__':
    main()
