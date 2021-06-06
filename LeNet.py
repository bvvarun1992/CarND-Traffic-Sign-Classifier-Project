#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeNet architecture 

@author: Varun Venkatesh
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.layers import flatten

def LeNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_weight = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean=mu, stddev=sigma))
    conv1_bias = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + conv1_bias

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_weight = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean=mu, stddev=sigma))
    conv2_bias = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(conv1, conv2_weight, strides=[1, 1, 1, 1], padding='VALID') + conv2_bias
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc_flat = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_weight = tf.Variable(tf.truncated_normal(shape= [400, 120], mean=mu, stddev=sigma))
    fc1_bias = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc_flat,fc1_weight), fc1_bias)
    
    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_weight = tf.Variable(tf.random_normal(shape= [120, 84], mean=mu, stddev=sigma))
    fc2_bias = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1,fc2_weight), fc2_bias)
    
    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_weight = tf.Variable(tf.random_normal(shape= [84, 43], mean=mu, stddev=sigma))
    fc3_bias = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2,fc3_weight), fc3_bias)
    
    return logits