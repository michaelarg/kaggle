import os,sys
import numpy as np
import tensorflow as tf
#import weakref

from tflearn.layers.conv import conv_2d, max_pool_2d
 

#def u_net():

#def variables)
X = tf.placeholder(tf.float32, shape=[None, 640, 960, 3], name="X")
y = tf.placeholder(tf.float32, shape=[None, 640, 960, 1], name="y")

a = tf.constant(2, name = 'start')
b=  tf.constant(3)
c = tf.add(a,b) 
#tf.scalar_summary("cost", c)

'''this function will be used to concatenate the matrices on the same row of the symmetric u pattern'''
def con_concat(matA , matB, filter_n , name):
 #  up_conv = upconv_2D(matA, n_filter, flags, name)
    return tf.concat([matA, matB ], axis = -1, name="concat_{}".format(name))                           #values,axis,name = 'concat'

def upconv_2D(tensor, n_filter, name):
    return tf.layers.conv2d_transpose(tensor,filters=n_filter,kernel_size=2,strides=2,name="upsample_{}".format(name))


'''
What does with tf.variable_scope mean? Often we want to share variables between values - for example if we have one image filter
function, and the filter is some matrix X. We want to multiply A by X and B by X. Instead of initialising the function twice and
having to store X essentially twice. We can share the X variable between A and B.

So instead we may create variables in a seperate bit of code and pass the to the functions that use them.
THis breaks encapsulation - the code that builds the graph must document the names, types, and shapes of variables to create.
when the code changes, the callers may have to create more or less, or different variables.

Alternatively - we use classes to create a model, where the classes take care of managing the variables they need.
WIthout classes tensorflow provides a variable scope mechanism that allows to easily share named variables while contructing a graph.

https://www.tensorflow.org/versions/r1.2/programmers_guide/variable_scope

- 
'''

filter_list=[64, 128, 256, 512,1024]


'''here we want to define what actually happens in the convolution'''
def conv2d_layers(input, filter_size, name , pool=True):
    tensor1 = input
    print "input tensor" , tensor1
    print "test1"
    with tf.variable_scope("layer_{}".format(name)):
#        print type(tensor1)
 #       print index
#        print filter_num
        tensor = tf.layers.conv2d(tensor1, filter_size , kernel_size = [3,3],padding = 'VALID' ,  activation = None, name = "conv1_layer{}".format(name))   
        print "test2"
        tensor  = tf.nn.relu(tensor ,name = "relu1_{}".format(name))

        tensor = tf.layers.conv2d(tensor, filter_size , kernel_size = [3,3],padding = 'VALID' ,  activation = None, name = "conv2_layer{}".format(name))  
        print "test2"
        tensor  = tf.nn.relu(tensor ,name = "relu2_{}".format(name))



        #possible batch normalisation layer
        if pool == True:
            pool = tf.layers.max_pooling2d(tensor, (2,2), strides = (2,2) , name="pool_{}".format(name))
 #           return pool
#        else:
#            return tensor
            return tensor, pool

        else:
            return tensor
            
       # enumerate.next()
#we actually can't throw away the original tensor after the pooling operation because we use it in the concat step

'''define the architecture'''

def maxpool_scale(convleft, upconvright):
    print "its working!"
#    print tensor.shape[1]
 #   print in_tensor.shape[1]
    #We are using zero padding and a stride of 1 so our equation becomes we have the current dimension W1 and W2:
    #W2 = (W1 - F + 2P)/S + 1 :
    #Becomes W2 = (W1 - F)/1 + 1
    #Need to solve this equation to get the size of the filter for the max pooling op to produce the right sized matrices
    a = convleft.shape[1]
    b = upconvright.shape[1]

    xx = a+1 - b 
    
    scale_conv4 = tf.layers.max_pooling2d(convleft, pool_size=(xx,xx), strides = 1)
    return scale_conv4

def unet_deconv(tensor, filter_size, name):
    tf.nn.conv2d_transpose(tensor, filters= filter_size, kernel_size = 2, strides = 2, name="upsample_{}".format(name))

def build_unet(input_tensor):
    #Left Side:
    conv1,pool1 = conv2d_layers(input_tensor , 64 , name = "conv1")
    conv2,pool2 = conv2d_layers(pool1 , 128 , name = "conv2")   
    conv3, pool3 = conv2d_layers(pool2 , 256 , name = "conv3")
    conv4, pool4 = conv2d_layers(pool3 , 512 , name = "conv4")
    conv5 = conv2d_layers(pool4 , 1024 , name = "conv5", pool = False)


    #Right Side:
    upconv6 = upconv_2D(conv5, 512 , name = "upconv6")
    scale_conv4 = maxpool_scale(conv4, upconv6)
    conv6 = con_concat(scale_conv4, upconv6, 1024, name = 'concat')
    conv6 = conv2d_layers(conv6 , 512 , name = "conv_up6", pool = False)

    upconv7 = upconv_2D(conv6, 256 , name = "upconv7")

    scale_conv3 = maxpool_scale(conv3, upconv7)
    conv7 = con_concat(scale_conv3, upconv7, 512, name = 'concat')
    conv7 = conv2d_layers(conv7 , 256 , name = "conv_up7", pool = False)


    upconv8 = upconv_2D(conv7, 128 , name = "upconv8")

    scale_conv2 = maxpool_scale(conv2, upconv8)
    conv8 = con_concat(scale_conv2, upconv8, 256, name = 'concat')
    conv8 = conv2d_layers(conv8, 128 , name = "conv_up8", pool = False)

    upconv9 = upconv_2D(conv8, 64 , name = "upconv9")
    scale_conv1 = maxpool_scale(conv1, upconv9)
    conv9 = con_concat(scale_conv1, upconv9, 128, name = 'concat')
    conv9 = conv2d_layers(conv9 , 64 , name = "conv_up9", pool = False)

    return tf.layers.conv2d(conv9, 2, (1,1), name='finaloutput', activation = tf.nn.sigmoid, padding = 'VALID')

''' intersection over union is a common metric for assessing performance in semantic segementations of tasls.
    Think about image segmentation.
    You want to identify Stop signs in your model. What you would do to train the model is to put a recetangle around the stop
    sign in the image.
    Then your model would attempt to do the same. Ideally your produced mask or rectangle would overlap the trained rectangle of
    the stop sign perfectly and receive an IOU of 1. If it overlaps it half the area we would receive an IOU of .5
    so area of intersection/area of two boxes
'''   

    
def iou(train , test):
    

    return value


def train_loss(xtrain, ytest):
    loss = iou(xtrain, ytest)
    global_setup = tf.train.get_or_create_global_step() #what is this? Returns and create (if necessary) the global step tensor.
    #global step tensor refers to the number of batches seen by the graph. 
    optimiser = tf.train.AdamOptimizer()
    return optimiser.minimize(loss, global_step = global_setup)



def main():    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 572, 572, 1], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 640, 960, 1], name="y")
    
    pred = build_unet(X)
    
    tf.add_to_collection("inputs", X)
    tf.add_to_collection("outputs", pred)

'''
What are the following lines doing? 
get collection - used for building graphs, returns a list of values in the collection with the given name
so then we ask what is tf.graphkeys.update_ops
tf.graphkeys -> the standard library uses various well known names to collect and retrieve values assosciated with a graph.

Basically says - update the moving averages before finishing teh training step

explained well here: http://ruishu.io/2016/12/27/batchnorm/
'''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    with tf.control_dependencies(update_ops):
        train_op = train_loss(pred , y)


    with tf.Session() as sess:
        writer = tf.summary.FileWriter('outputs', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        #print sess.run(c)
        #print 'printing in the session' , sess.run(X)
     #   writer = tf.summary.FileWriter('./graphs', sess.graph)
     #   writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
     
        writer.close()

if __name__ == '__main__':
    main()

#with tf.Session() as sess:
#	print(sess.run(h))
#up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)

'''
l: feature maps
k: feature maps as outputs
n * m : filter size
k = number of filters
f = filter spatial extent
s = stride
p = padding
w1 = width
h1 = height
d1 = dimension of input
Example with MNIST:
input image: 1*28*28
convd with 5*5 filter size stride of 1 and padding 

w2= (w1- f + 2p)/s + 1
h2 = (h1 - f + 2ps)/s + 1
d2 = k

w2 = (28- (5-1)) = 24
h2 = (28- (5-1)) = 24
d2 = 32
thus 32X24X24 -> weights(F*F*D1) * K weights + K biases (5*5*1+1) * 32 = 832 parameters

maxpool1 = 2x2 window is replaced with max value  w2 = (w1-F)/s + 1 -> (28-5)/2 + 1 
        =  32X12X12

conv2d2 = (12-(3-1))=10 -> 32X10X10 -> weights(F*F*D1) * K weights + K biases (3*3*32+1) * 32 = 9248 parameters

maxpool2 = 32X5X5


572X572
p=0
s =2 
3X3 filter
w2 = (572 - (3-1)) = 570
h2 = (572 - (3-1)) = 570
d2 = 64

w2= (w1- f + 2p)/s + 1
h2 = (h1 - f + 2ps)/s + 1
d2 = k
similarly for the third

max pool -> (568-2)/2 + 1  = 283 + 1 = 284

cov layer = (284 - (3-1)) = 282
cov layer = 280

max pool -> (280-2)/2 + 1 = 140X140
conv = 138
conv = 136
max pool = 68X68     ----- filters number also doubling
conv = 66
conv 64
max pool = 32
conv = 30
conv = 28X28
-------- upsampling-decoding ------------


input image 

'''
