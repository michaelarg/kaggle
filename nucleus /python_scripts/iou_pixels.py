import os,sys
from pixeltoform import plot_test
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import matplotlib.pyplot as plt

'''
23/2/18 - MA
The last  code dealt with rectangle masks, this will be a per pixel mask. Can assume black background.

How it works - calculates the non zero pixels in the image which is not going to work with real image segemntation

Input: two images - when actually using the algorithm one will be the train image with a box around the
area of interest. The other will be the predicted box.

Output: iou metric as a float and possibly the image of the two boxes drawn.

*Issues - right now the problem is that it is a solid box and the whole algorithm relies on it being
a solid box. What happens when the box is not filled in and there is an image in there? Probably fails.


*Todo - one remedy is to have the algoritm
'''


def draw_img(iimg):
 #  img = np.zeros([150,150])

#    cv2.rectangle(img,(ix1,iy1),(ix2,iy2),(0,0,0), thickness = -1) #top left and bottom right 
    cv2.imshow('image',iimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlap(img,img2):
    
    kk = img.flatten()
    area1 = np.count_nonzero(kk) #just total number of non zero pixels is the area


    kp = img2.flatten()
    area2 = np.count_nonzero(kp)
    print area1
    print area2
 #   diff = kk - kp
#    print type(kp)
#    print np.count_nonzero(diff)

#    j = kk^kp
    cc = np.bitwise_and(kk,kp)
    b = np.count_nonzero(cc)

 
    numerator = b

    denominator = (area1 + area2)-numerator

    return float(numerator)/float(denominator)

def main():
    os.chdir("/Users/michaelargyrides/Desktop/iou_test")
    img = cv2.imread('test1.png') #the image read in is a grayscale image
    img2= cv2.imread('test2.png')

    draw_img(img)
    draw_img(img2)
    
    x1 = overlap(img,img2)
    print x1
 #   draw_box(img,img2, img,img2)

#    print iou_metric(box_shape(img),box_shape(img2),overlap(img,img2))

if __name__ == '__main__':
    main()
