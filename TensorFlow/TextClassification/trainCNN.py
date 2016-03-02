import numpy as np 
import tensorflow as tf 
import os 
import time 
import datetime
from sklearn.cross_validation import train_test_split

import data_helpers2
from convNN import TextCNN

# Parameters

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")


tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

print('-----Parameters-----')
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print('{} = {}'.format(attr.upper(), value))
print('\n')


# Get Data
X, y, vocabulary, vocabulary_inv = data_helpers2.load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print 'Vocabulary Size: {}'.format(len(vocabulary))
print 'Train / Test Split: {} / {}'.format(X_train.shape[0], X_test.shape[0])

# Train

with tf.Graph().as_default():
	session_config = tf.ConfigProto(
		allow_soft_placement = FLAGS.allow_soft_placement
		log_device_placement = FLAGS.log_device_placement)
	
	sess = tf.Session(config = session_config)

	with sess.as_default():

		cnn = TextCNN(
            sequence_length = X_train.shape[1],
            num_classes = 2,
            vocab_size = len(vocabulary),
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = map(int, FLAGS.filter_sizes.split(",")),
            num_filters = FLAGS.num_filters,
            l2_reg_lambda = FLAGS.l2_reg_lambda)


