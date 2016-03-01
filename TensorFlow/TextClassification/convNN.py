import numpy as np 
import tensorflow as tf 

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
    initial = tf.constant(constant, shape = shape)
    return tf.Variable(initial)


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
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(W, self.x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
        # convolution layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv {}'.format(filter_size)):
                # conv
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = weight_variable(filter_shape)
                b = bias_variable([num_filters])
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides = [1, 1, 1, 1], padding = 'VALID')
                hidden = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(hidden, ksize = [1, sequence_length - filter_size + 1, 1, 1],
                                       strides = [1, 1, 1, 1], padding = 'VALID')
                pooled_outputs.append(pooled)
                
        # combine all pooled
        num_filters_total = num_filters * len(filter_sizes)
        self.hidden_pool = tf.concat(3, pooled_outputs)
        self.hidden_pool_flat = tf.reshape(self.hidden_pool, [-1, num_filters_total])
        
        # dropout
        with tf.name_scope('dropout'):
            self.hidden_drop = tf.nn.dropout(self.hidden_pool_flat, self.keep_prob)
            
        # output layer
        with tf.name_scope('output'):
            W = weight_variable([num_filters_total, num_classes])
            b = bias_variable([num_classes])
            reg = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            l2_loss += reg
            self.scores = tf.nn.xw_plus_b(self.hidden_drop, W, b)
            self.prediction = tf.argmax(self.scores, 1)
            
        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y_)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            
        # evaluate
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.prediction, tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            