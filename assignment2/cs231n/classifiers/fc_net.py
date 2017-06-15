from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.D = input_dim
        self.M = hidden_dim
        self.C = num_classes
        self.reg = reg

        w1 = np.random.normal(0, weight_scale, [self.D, self.M] )
        b1 = np.zeros(self.M)
        w2 = np.random.normal(0, weight_scale,  [self.M, self.C] )
        b2 = np.zeros(self.C)

        self.params.update({'W1': w1,'W2': w2, 'b1': b1, 'b2': b2})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']

        # Forward_1st_layer:
        layer1_out, cache_layer1 = affine_relu_forward(X, W1, b1)

        # Forward_2nd_layer:
        layer2_out, cache_layer2 = affine_forward(layer1_out, W2, b2)

        scores = layer2_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        soft_loss, dscores = softmax_loss(scores, y)
        regular_1 = 0.5 * self.reg * np.sum(W1**2)
        regular_2 = 0.5 * self.reg * np.sum(W2**2)
        loss = soft_loss + regular_1 + regular_2


        # Backpropagaton
        # Backprop into second layer
        dx2, dW2, db2 = affine_backward(dscores, cache_layer2)
        dW2 += self.reg * W2

        # Backprop into first layer
        dx1, dW1, db1 = affine_relu_backward( dx2, cache_layer1)
        dW1 += self.reg * W1

        grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        dimension_temp = [input_dim] + hidden_dims + [num_classes]

        for i in range(0, self.num_layers):
            W_each_layer = 'W' + str(i+1)
            b_each_layer = 'b' + str(i+1)

            if use_batchnorm and i != (self.num_layers-1):
                gamma_each_layer = 'gamma' + str(i+1)
                beta_each_layer  = 'beta'  + str(i+1)
                self.params[gamma_each_layer] = np.ones(dimension_temp[i])
                self.params[beta_each_layer]  = np.zeros(dimension_temp[i])

            self.params[b_each_layer] = np.zeros(dimension_temp[i+1])  ## Dimesion of Class: C
            self.params[W_each_layer] = np.random.normal(scale=weight_scale, 
                size=(dimension_temp[i], dimension_temp[i+1]))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        #print (X.shape)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        self.cache = {}
        self.dropout_cache = {}
        self.batchnorm_cache = {}
        N = X.shape[0]
        scores = X.reshape(N,-1)
        

        for i in range(1, self.num_layers+1):
          W_each_layer      = 'W' + str(i)
          b_each_layer      = 'b' + str(i)
          gamma_each_layer  = 'gamma' + str(i)
          beta_each_layer   = 'beta' + str(i)
          bn_each_layer     = 'batchnorm' + str(i)
          dropout_each_layer = 'dropout' + str(i)
          cache_each_layer  = 'c' + str(i)
        
        ## For Last Layer
          if i == self.num_layers: 
            scores, cache = affine_forward(scores, self.params[W_each_layer], self.params[b_each_layer])
        
        ## For the rest of Layers  
          else:
            if self.use_batchnorm:
                #print (scores.shape)
                scores, self.batchnorm_cache[bn_each_layer] = batchnorm_forward(scores, 
                    self.params[gamma_each_layer], 
                    self.params[beta_each_layer], 
                    self.bn_params[i-1])

            ## affine_relu_forward (x, W, b)
            scores, cache = affine_relu_forward(scores, self.params[W_each_layer], self.params[b_each_layer])

            if self.use_dropout:
              scores, self.dropout_cache[dropout_each_layer] = dropout_forward(scores, self.dropout_param)

          self.cache[cache_each_layer] = cache
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        ## compute the initial loss (without L2 regularization) from last layers
        loss, dout = softmax_loss(scores, y)
 
        ## Compute the loss and gredient 
        for i in range(self.num_layers, 0, -1):
          W_each_layer      = 'W' + str(i)
          b_each_layer      = 'b' + str(i)
          gamma_each_layer  = 'gamma' + str(i)
          beta_each_layer   = 'beta' + str(i)
          bn_each_layer     = 'batchnorm' + str(i)
          dropout_each_layer = 'dropout' + str(i)
          cache_each_layer   = 'c' + str(i)

          ## ADD L2 regularization term to Loss
          loss += 0.5*self.reg*np.sum(self.params[W_each_layer]**2) 
        
        ## For Last Layer  
          if i == self.num_layers:
            dout, grads[W_each_layer], grads[b_each_layer] = affine_backward(dout, self.cache[cache_each_layer])
            """
            affine_backward (dout, cache)
            Inputs:
            - dout: Upstream derivativative, of shape (N, M)
            - cache: Tuple of:
                - xi: Input data, of shape (N,D_1,...,D_k) 
                - wi: Weights, of shape (D, M)
            """
        ## For the rest of Layer
          else:
            if self.use_dropout:
              dout = dropout_backward(dout, self.dropout_cache[dropout_each_layer])

            dout, grads[W_each_layer], grads[b_each_layer] = affine_relu_backward(dout, self.cache[cache_each_layer])

            if self.use_batchnorm:
              dout, grads[gamma_each_layer], grads[beta_each_layer] = batchnorm_backward(dout, self.batchnorm_cache[bn_each_layer])


        ## Add Derivativ of L2 regularization term d(0.5*reg*Wi**2) == 0.5*2*reg*Wi
          grads[W_each_layer] += self.reg*self.params[W_each_layer]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
