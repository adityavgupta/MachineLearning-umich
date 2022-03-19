import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:

  conv - relu - 2x2 max pool - fc - relu - fc - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3,
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
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights for the convolutional layer using the keys 'W1' (here      #
    # we do not consider the bias term in the convolutional layer);            #
    # use keys 'W2' and 'b2' for the weights and biases of the                 #
    # hidden fully-connected layer, and keys 'W3' and 'b3' for the weights     #
    # and biases of the output affine layer. For this question, we assume      #
    # the max-pooling layer is 2x2 with stride 2. Then you can calculate the   #
    # shape of features input into the hidden fully-connected layer, in terms  #
    # of the input dimension and size of filter.                               #
    ############################################################################
    C, H, W = input_dim
    hH = 1+(H - filter_size) // 2
    hW = 1+(W - filter_size) // 2
    self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters,C, filter_size, filter_size))
    self.params['W2'] = np.random.normal(scale = weight_scale, size = (num_filters * hH * hW, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(loc=0.0, scale = weight_scale, size = (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_c = conv_forward(X,W1)
    r1_o, r1_c = relu_forward(conv_out)
    p_o, p_c = max_pool_forward(r1_o, pool_param)
    fc1_o, fc1_c = fc_forward(p_o.reshape(p_o.shape[0], p_o.shape[1]*p_o.shape[2]*p_o.shape[3]), W2, b2)
    r2_o, r2_c = relu_forward(fc1_o)
    scores, fc2_c = fc_forward(r2_o, W3, b3) 
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
    # for self.params[k].                                                      #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    dx, grads['W3'], grads['b3'] = fc_backward(dx, fc2_c)
    dx = relu_backward(dx, r2_c)
    dx, grads['W2'], grads['b2'] = fc_backward(dx, fc1_c)
    #dx = dx.reshape(p_c[0].shape[0], p_c[0].shape[1], p_c[0].shape[2]//2, p_c[0].shape[3]//2)
    dx = max_pool_backward(dx.reshape(p_o.shape), p_c)
    dx = relu_backward(dx, r1_c)
    dx, grads['W1'] = conv_backward(dx, conv_c)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
