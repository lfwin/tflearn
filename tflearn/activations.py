# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

import tflearn
from . import initializations
from . import variables as va
from .utils import get_from_module, get_incoming_shape
import math

def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'activation')


""" Activation Functions """

def stagger(x):
    """
    -0.01 =< x <= 0.01, return 0
    0.01 =< x , return x - 0.01
    x <= -0.01, return x + 0.01
    """
    y = tf.maximum(x - 0.01, 0)
    z = tf.minimum(x + 0.01, 0)
    return tf.select(tf.greater_equal(x, 0), y, z)

def bi_relu(x):
    """
    return [relu(x), relu(-x)]
    """
    res = tf.concat(3, [tf.nn.relu(x), tf.nn.relu(tf.neg(x))])
    #print(tf.shape(res))
    
    return res

def reverse_relu(x):

    return tf.nn.relu(tf.neg(x))


def level_relu(x, alpha = -2.5):
    y = tf.mul(x, tf.to_float(tf.greater_equal(x, alpha)))
    z = tf.mul(x, tf.to_float(tf.less(x, alpha)))
    return tf.concat(3, [y, z])

            
def BiBU(x):
    """
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    Binary Branch Units
    """
    input_shape = get_incoming_shape(x)
    binary_x = tf.sign(x)
    forward = tf.concat(len(input_shape)-1, [tf.nn.relu(binary_x),  tf.neg(tf.nn.relu(tf.neg(binary_x)))])
    backward = tf.concat(len(input_shape)-1, [tf.nn.relu(x),  tf.neg(tf.nn.relu(tf.neg(x)))])
    
    return backward + tf.stop_gradient(forward - backward)
    

def twins_relu(x):
    """
    return [relu(x), -relu(-x)]
    """
    from tflearn import utils
    #from tflearn.layers.conv import max_pool_2d
    input_shape = get_incoming_shape(x)
    res = tf.concat(len(input_shape)-1, [tf.nn.relu(x),  tf.neg(tf.nn.relu(tf.neg(x)))])
    
    #print(tf.shape(res))
    
    return res

def abs(x):
    
    return tf.abs(x)

def linear(x):
    """ Linear.

    f(x) = x

    Arguments:
        x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.

    Returns:
        The incoming Tensor (without changes).
    """
    return x


def tanh(x):
    """ Tanh.

    Computes hyperbolic tangent of `x` element-wise.

    Arguments:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
            or `qint32`.

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
          the return type is `quint8`.
    """
    return tf.tanh(x)


def sigmoid(x):
    """ Sigmoid.

    Computes sigmoid of `x` element-wise.
    Specifically, `y = 1 / (1 + exp(-x))`.

    Arguments:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
            or `qint32`.

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
        the return type is `quint8`.
    """
    return tf.nn.sigmoid(x)


def softmax(x):
    """ Softmax.

    Computes softmax activations.

    For each batch `i` and class `j` we have

      softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`. 2-D with shape `[batch_size, num_classes]`.

    Returns:
        A `Tensor`. Has the same type as `x`. Same shape as `x`.
    """
    return tf.nn.softmax(x)


def softplus(x):
    """ Softplus.

    Computes softplus: `log(exp(features) + 1)`.

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`.

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return tf.nn.softplus(x)


def softsign(x):
    """ Softsign.

    Computes softsign: `features / (abs(features) + 1)`.

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`.

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return tf.nn.softsign(x)


def relu(x):
    """ ReLU.

    Computes rectified linear: `max(features, 0)`.

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`.

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return tf.nn.relu(x)


def relu6(x):
    """ ReLU6.

    Computes Rectified Linear 6: `min(max(features, 0), 6)`.

    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.

    Returns:
        A `Tensor` with the same type as `x`.
    """
    return tf.nn.relu6(x)


def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
    """ LeakyReLU.

    Modified version of ReLU, introducing a nonzero gradient for negative
    input.

    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        alpha: `float`. slope.
        name: A name for this activation op (optional).

    Returns:
        A `Tensor` with the same type as `x`.

    References:
        Rectifier Nonlinearities Improve Neural Network Acoustic Models,
        Maas et al. (2013).

    Links:
        [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf]
        (http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

    """

    # If incoming Tensor has a scope, this op is defined inside it
    i_scope = ""
    #if hasattr(x, 'scope'):
        #if x.scope: i_scope = x.scope
    with tf.name_scope(i_scope + name) as scope:
        x = tf.nn.relu(x)
        m_x = tf.nn.relu(-x)
        x -= alpha * m_x

    x.scope = scope

    return x

# Shortcut
leakyrelu = leaky_relu


def prelu(x, weights_init='zeros', restore=True, name="PReLU"):
    """ PReLU.

    Parametric Rectified Linear Unit.

    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        weights_init: `str`. Weights initialization. Default: zeros.
        restore: `bool`. Restore or not alphas
        name: A name for this activation op (optional).

    Attributes:
        scope: `str`. This op scope.
        alphas: `Variable`. PReLU alphas.

    Returns:
        A `Tensor` with the same type as `x`.

    References:
        Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification. He et al., 2014.

    Links:
        [http://arxiv.org/pdf/1502.01852v1.pdf]
        (http://arxiv.org/pdf/1502.01852v1.pdf)

    """
    w_shape = tflearn.utils.get_incoming_shape(x)[1:]

    # If incoming Tensor has a scope, this op is defined inside it
    i_scope = ""
    if hasattr(x, 'scope'):
        if x.scope: i_scope = x.scope
    with tf.name_scope(i_scope + name) as scope:
        W_init = initializations.get(weights_init)()
        alphas = va.variable(shape=w_shape, initializer=W_init,
                             restore=restore, name=scope + "alphas")

        x = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5

    x.scope = scope
    x.alphas = alphas

    return x


def elu(x):
    """ ELU.

    Exponential Linear Unit.

    Arguments:
        x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        name : A name for this activation op (optional).

    Returns:
        A `tuple` of `tf.Tensor`. This layer inference, i.e. output Tensors
        at training and testing time.

    References:
        Fast and Accurate Deep Network Learning by Exponential Linear Units,
        Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter. 2015.

    Links:
        [http://arxiv.org/abs/1511.07289](http://arxiv.org/abs/1511.07289)

    """

    return tf.nn.elu(x)
