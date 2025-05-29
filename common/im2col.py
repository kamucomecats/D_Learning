import numpy as np
from numpy.lib.stride_tricks import as_strided

#assumed input shape is [batch_size, channel, height, width]
#assumed output shape is [batch_size, -1] with pad
def im2col(X, FH, FW, stride=1, pad=0): #4-dim input intended

    X = pad_input(X, pad) #padding here
    print(X.shape)
    N, C, H, W = X.shape
    OH = (H - FH) // stride + 1
    OW = (W - FW) // stride + 1
    
    print(X.strides)
    s0, s1, s2, s3 = X.strides
    shape = (N, C, OH, OW, FH, FW)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    
    patches = as_strided(X, shape=shape, strides=strides)
    return patches.reshape(N * OH * OW, C * FH * FW) #per receptive-area

def pad_input(x, pad):
    N, C, H, W = x.shape
    H_pad = H + pad * 2
    W_pad = W + pad * 2
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype=x.dtype)
    
    x_pad[:, :, pad:pad+H, pad:pad+W] = x
    print(x_pad)
    return x_pad

x = np.array([[[[1.0, 2.0, 7.0],
     [3.0, 5.0, 3.0],
     [4.0, 8.0, 3.0],]],
    [[[7.0, 2.0, 7.0],
     [3.0, 5.0, 3.0],
     [4.0, 8.0, 3.0],]]])

print(x)
print(x.shape)
x = im2col(x, FH=2, FW=2)
print(x)
print(x.shape)