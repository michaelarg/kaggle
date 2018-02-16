import os,sys
import collections
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import re
#path_local = "/Users/michaelargyrides/Documents/datascience_nuclei/train_subset"
#os.chdir(path_local)
os.chdir("/Users/michaelargyrides/Documents/datascience_nuclei/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks")
img = cv2.imread('7a7684685d8b8263a84a9fcd3fcfb05f35f4b1b374d6a0ed87fa5df752dcb8fd.png',0)
img2 = cv2.imread('7a7684685d8b8263a84a9fcd3fcfb05f35f4b1b374d6a0ed87fa5df752dcb8fd.png',0)

print np.shape(img)

#
def iou(img, img2):
    print 'img non zero cont', np.count_nonzero(img)
    print 'img2 non zero cont', np.count_nonzero(img2)
    
    img = img.flatten()
    img2 = img2.flatten()

        
    return img+img2

def main():
    a = iou(img2,img2)
    print np.shape(a)


if __name__ == '__main__' :
    main()
