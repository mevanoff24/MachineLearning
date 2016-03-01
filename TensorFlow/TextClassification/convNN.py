import numpy as np 
import tensorflow as tf 


class TextCNN(object):

	def __init__(self, sequence_length, num_classes, vocab_size,
		embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		# placeholders
		self.x = tf.placeholder(tf.int32, [None, sequence_length])
		sef.y_ = tf.placeholder(tf.int32, [None, num_classes])
		self.keep_prob = tf.placeholder(tf.float32)
		# l2 regularization
		l2_loss = tf.constant(0.0)

		# embedding layer
		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size]))