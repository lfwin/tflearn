# -*- coding: utf-8 -*-

""" Wide Residual Network.

Applying a Wide Residual Network to CIFAR-100 Dataset classification task.

References:
    - Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
    - wide Residual Network
Links:
    - [wide Residual Network]
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
from PIL import Image
from tflearn.layers.conv import conv_2d, max_pool_2d
import tensorflow as tf


depth = 16              # table 5 on page 8 indicates best value (4.17) CIFAR-10
k = 2                  # 'widen_factor'; table 5 on page 8 indicates best value (4.17) CIFAR-10
dropout_probability = 0 # table 6 on page 10 indicates best value (4.17) CIFAR-10

#weight_decay = 0.0005   # page 10: "Used in all experiments"

# Data loading
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()

Y = tflearn.data_utils.to_categorical(Y, 10)
testY = tflearn.data_utils.to_categorical(testY, 10)
# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
img_prep.add_featurewise_stdnorm(per_channel=True)
img_prep.add_zca_whitening()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride, act = "twins_relu"):
    def f(net, scope=None, reuse=False, name="WSN"):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3,3,stride,"same"],
                        [3,3,(1,1),"same"] ]
        
        n_bottleneck_plane = n_output_plane
        #res = net
        with tf.variable_op_scope([net], scope, name, reuse=reuse) as scope:
            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = tflearn.batch_normalization(net)
                        net = tflearn.activation(net, act)
                        convs = net
                    else:
                        convs = tflearn.batch_normalization(net)
                        convs = tflearn.activation(convs, act)
                    convs = conv_2d(convs, n_bottleneck_plane, 3, strides= v[2], activation='linear', 
                                    weights_init='he', bias=True, regularizer='L2', weight_decay=0.0001)
                else:
                    convs = tflearn.batch_normalization(convs)
                    convs = tflearn.activation(convs, act)
                    if dropout_probability > 0:
                        convs = tflearn.dropout(convs, dropout_probability)
                    convs = conv_2d(convs, n_bottleneck_plane, 3, strides= v[2], activation='linear', 
                                    weights_init='he',bias=True, regularizer='L2', weight_decay=0.0001)

            if n_input_plane != n_output_plane:
                shortcut = conv_2d(net, n_output_plane, 1, strides= stride, activation='linear', 
                                   weights_init='he',bias=True, regularizer='L2',weight_decay=0.0001)
            else:
                shortcut = net
            
            res = tf.add(convs, shortcut)

        return res
    
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2,int(count+1)):
            net = block(n_output_plane, n_output_plane, stride=(1,1))(net)
        return net
    
    return f

def create_model():
    # Building Wide Residual Network
    
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    n_stages=[16, 16*k, 32*k, 64*k]

    net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    conv1 = conv_2d(net, n_stages[0], 3, activation='linear', bias=False, 
                    regularizer='L2', weight_decay=0.0001)

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"

    net = tflearn.batch_normalization(conv4)
    net = tflearn.activation(net, 'twins_relu')
    net = tflearn.avg_pool_2d(net, 8)
    #net = tflearn.avg_pool_2d(net, kernel_size=8, strides=1, padding='same')
    net = tflearn.fully_connected(net, 10, activation='softmax')
    
    return net

if __name__ == '__main__':
    net = create_model()
    mom = tflearn.Momentum(0.0001, lr_decay=0.1, decay_step=1000000, staircase=True)
    net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=1)
    
    model.load("/home/lfwin/my_tmp/tflearn_logs/cifar10_WRN2_tReLU_130/model.tfl")
    
    model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(testX, testY),
          show_metric=True, batch_size=32, run_id='cifar10_WRN2_tReLU_140')
    
    # Manually save model
    model.save("/home/lfwin/my_tmp/tflearn_logs/cifar10_WRN2_tReLU_140/model.tfl")
    
    