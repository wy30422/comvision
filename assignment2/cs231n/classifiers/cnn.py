from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        
        F = num_filters
        C, H, W = input_dim
        
        self.params['W1'] = weight_scale * np.random.randn(F,C,filter_size,filter_size)
        self.params['b1'] = np.zeros(F)

        ## Padding
        P = (filter_size - 1) / 2
        S_conv = 1
        H_pad = (H + 2 * P - filter_size) / S_conv + 1
        W_pad = (W + 2 * P - filter_size) /  S_conv + 1

        ## Pool
        W_pool = 2
        H_pool = 2
        S_pool = 2
        Hp = (H_pad - H_pool) / S_pool + 1
        Wp = (W_pad - W_pool) / S_pool + 1


        self.params['W2'] = weight_scale * np.random.randn(F * Hp * Wp, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)


        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        a1, cache1= conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        a1_shape_convert = a1.reshape(a1.shape[0], -1)
        a2, cache2 = affine_relu_forward(a1_shape_convert, W2, b2)
        scores, cache3 = affine_forward(a2, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss_temp, dscores = softmax_loss(scores, y)
        loss = loss_temp + 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        da2, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
        grads['W3'] += self.reg * cache3[1]
        da1, grads['W2'], grads['b2'] = affine_relu_backward(da2, cache2)
        grads['W2'] += self.reg * cache2[0][1]
        da1_reshape = da1.reshape(*a1.shape)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(da1_reshape, cache1)
        grads['W1'] += self.reg * cache1[0][1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads



