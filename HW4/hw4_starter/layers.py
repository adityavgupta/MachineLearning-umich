from builtins import range
from multiprocessing import pool
from xml.etree.ElementTree import indent
import numpy as np
import math
import scipy.signal


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    out = np.dot(x,w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dw = x.T @ dout
    db = np.ones(dout.shape[0])@dout
    dx = dout@w.T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    relu = lambda x : x*(x > 0).astype(float)
    out = relu(x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout*(x>=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def index_generator(C, H, W, H_p, W_p, stride=1):
  HH = (H- H_p)//stride + 1
  WW = (W-W_p)//stride + 1
  r1 = np.arange(W_p)
  R = np.tile(r1, H_p).reshape(H_p,W_p)
  R += W * np.arange(H_p).reshape(-1,1)
  R = R.flatten()

  RGB = np.tile(R,C)
  RGB = np.reshape(RGB,(C,H_p,W_p))
  RGB += H*W * np.arange(C).reshape(C,-1,1)

  c1 = RGB.flatten()

  HHWW = HH*WW
  ind = np.tile(c1,WW).reshape(WW,-1)
  ind += stride*np.arange(WW).reshape(-1,1)
  ind = ind.reshape(-1)
  ind = np.tile(ind,HH)
  ind = ind.reshape(HH,-1)
  ind += stride*W * np.arange(HH).reshape(-1,1)
  ind = ind.flatten()
  ind = ind.reshape(HHWW, H_p*W_p*C)
  return ind

def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # given in Q1(c). Note that, as specified in the question, this function  #
    # should actually implement the filtering operation with w. This is just  #
    # equivalent to implementing convolution with a flipped filter, and this  #
    # can be compensated for in the gradient computation as you saw in the    #
    # derivation for Q1 (c).                                                  #
    #                                                                         #
    # Note: You are free to use scipy.signal.convolve2d, but we encourage you #
    # to implement the convolution operation by yourself using just numpy     #
    # operations.                                                             #
    ###########################################################################
    N, C, H, W = x.shape
    F, _, H_p, W_p = w.shape
    HH =  H - H_p + 1
    WW = W - W_p + 1
    ind = index_generator(C, H, W, H_p, W_p, stride=1)
    
    flt_img = x.reshape((N, C*H*W))
    conv_points  = flt_img[:,ind.flatten()].reshape((N,C*H_p*W_p, HH*WW), order='F')
    vec_wt = w.reshape((F, C*H_p*W_p)).T
    out = conv_points.transpose(0,2,1) @ vec_wt
    out = out.transpose(0,2,1).reshape((N, F, HH, WW))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    x, w = cache
    ###########################################################################
    # TODO: Implement the convolutional backward pass as defined in Q1(c)     #
    #                                                                         #
    # Note: You are free to use scipy.signal.convolve2d, but we encourage you #
    # to implement the convolution operation by yourself using just numpy     #
    # operations.                                                             #
    ###########################################################################
    _,_, hp, wp = w.shape
    dout_p = np.pad(dout, ((0,0),(0,0),(hp-1,hp-1),(wp-1,wp-1)), mode='constant',constant_values=((0,0),(0,0),(0,0), (0,0)))
    dx, _ = conv_forward(dout_p, w[:,:,::-1,::-1].transpose(1,0,2,3))
    dw, _ = conv_forward(x.transpose(1,0,2,3),dout.transpose(1,0,2,3))
    dw = dw.transpose(1,0,2,3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here and we can assume that the dimension of
    input and stride will not cause problem here. Output size is given by
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    s = pool_param['stride']

    H_prime = 1 + (H - ph) // s
    W_prime = 1 + (W - pw) // s

    ind = index_generator(C=1, H=H, W=W, H_p=ph, W_p=pw, stride=s)
    flt_x = x.reshape((N, C, H*W))
    extract = flt_x[:,:,ind]
    out = np.max(extract, axis=3, keepdims=True)
    out = out.reshape(N,C,H_prime,W_prime)

    stacked_ind = np.repeat(ind[np.newaxis,...], C, axis=0)
    stacked_ind = np.repeat(stacked_ind[np.newaxis,...],N,axis=0)
    stacked_ind = np.expand_dims(stacked_ind, axis=3)
    max_locs_ext = np.expand_dims(np.argmax(extract, axis=3),-1)

    onehot = (np.arange(ph*pw) == max_locs_ext[...,None]).astype(int)
    im_locs = (stacked_ind @ onehot.transpose(0,1,2,4,3)).squeeze()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, im_locs)
    return out, cache



def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param, im_locs = cache
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    s = pool_param['stride']

    dx = np.zeros(x.shape)
    N, C, H, W = x.shape
    _,_, HH, WW = dout.shape

    
    dYdX = (np.arange(H*W) == im_locs[...,None]).astype(int)
    dx = dout.reshape(N,C,1,dout.shape[2]*dout.shape[3]) @ dYdX
    dx = dx.reshape(N,C,H,W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the j-th
      class for the i-th input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the cross-entropy loss averaged over N samples.
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Implement the softmax loss
    ###########################################################################
    N = len(y)
    softmax = lambda a : np.exp(a)/np.sum(np.exp(a), axis=1, keepdims=True)
    y_hat = softmax(x)
    log_prob = np.log(y_hat[range(N), y])
    loss = -np.sum(log_prob)/N
    dx = y_hat.copy()
    dx[range(N), y] -=1
    dx = dx/N
    # dx = (y_hat[range(len(y_hat)), y])/len(y_hat)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx



