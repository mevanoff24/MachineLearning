import tensorflow as tf
import numpy as np
import data.image_loader

###############################
##### Set hyperparameters #####
###############################

num_epochs = 2
width = 150
height = 150
num_categories = 2
num_channels = 3
batch_size = 100 # for my sanity
num_training_examples = 2000
num_test_examples = 200

num_batches = num_training_examples/batch_size

####################################################################################
##### It's convenient to define some methods to perform frequent routine tasks #####
####################################################################################

def weight_variable(shape):
    '''
    Generates a TensorFlow Tensor. This Tensor gets initialized with values sampled from the truncated normal
    distribution. Its purpose will be to store model parameters.
    :param shape: The dimensions of the desired Tensor
    :return: The initialized Tensor
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Generates a TensorFlow Tensor. This Tensor gets initialized with values sampled from <some?> distribution.
    Its purpose will be to store bias values.
    :param shape: The dimensions of the desired Tensor
    :return: The initialized Tensor
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Generates a conv2d TensorFlow Op. This Op flattens the weight matrix (filter) down to 2D, then "strides" across the
    input Tensor x, selecting windows/patches. For each little_patch, the Op performs a right multiply:
            W . little_patch
    and stores the result in the output layer of feature maps.
    :param x: a minibatch of images with dimensions [batch_size, height, width, 3]
    :param W: a "filter" with dimensions [window_height, window_width, input_channels, output_channels]
    e.g. for the first conv layer:
          input_channels = 3 (RGB)
          output_channels = number_of_desired_feature_maps
    :return: A TensorFlow Op that convolves the input x with the filter W.
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    Genarates a max-pool TensorFlow Op. This Op "strides" a window across the input x. In each window, the maximum value
    is selected and chosen to represent that region in the output Tensor. Hence the size/dimensionality of the problem
    is reduced.
    :param x: A Tensor with dimensions [batch_size, height, width, 3]
    :return: A TensorFlow Op that max-pools the input Tensor, x.
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


############################
##### Set up the model #####
############################

x = tf.placeholder("float", shape=[None, height, width, num_channels])
x_image = tf.reshape(x, [-1, width, height, num_channels])
y_ = tf.placeholder("float", shape=[None, num_categories])

#1st conv layer
W_conv1 = weight_variable([5, 5, num_channels, 32]) #5x5 conv window, 3 colour channels, 32 outputted feature maps
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#fully connected layer
W_fc1 = weight_variable([38 * 38 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 38*38*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#droupout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax output layer
W_fc2 = weight_variable([1024, num_categories])
b_fc2 = bias_variable([num_categories])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#saving model
saver = tf.train.Saver()

###################################
##### Load data from the disk #####
###################################

dataset = data.image_loader.ImageLoad(base_path="/home/hal9000/Datasets/id_dataset3",
                                      num_categories=num_categories,
                                      width=width,
                                      height=height)

data_training = np.asarray(np.split(dataset.data_training, num_batches))
labels_training = np.asarray(np.split(dataset.labels_training, num_batches))

data_test = np.split(dataset.data_test, 1)
labels_test = np.split(dataset.labels_test, 1)

####################################################
##### Train the model and evaluate performance #####
####################################################

with tf.Session() as sess:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for j in range(num_epochs):
        for i in range(num_batches):
            train_step.run(feed_dict={x: np.asarray(data_training[i]), y_: np.asarray(labels_training[i]), keep_prob: 0.5})

        print "=== EPOCH: " + str(j) + " ==="
        print "test accuracy: %g \n"%accuracy.eval(feed_dict={x: data_test[i], y_: labels_test[i], keep_prob: 1.0})

    saver.save(sess, "saved_models/convnet_image" + str(j) + ".ckpt")