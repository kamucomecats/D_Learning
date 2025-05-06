import numpy as np
from numpy.lib.stride_tricks import as_strided

#assumed input shape is [batch_size, channel, height, width]
#assumed output shape is [batch_size, -1] with pad
def im2col(X, FH, FW, strides=1, pad=0):
    H, W = X.shape
    OH = H - FH + 1
    OW = W - FW + 1
    shape = (OH, OW, FH, FW)
    strides = X.strides * 2
    
    patches = as_strided(X, shape=shape, strides=strides)
    return patches.reshape(OH * OW, -1)

x = np.array([[1.0, 2.0, 7.0],
     [3.0, 5.0, 3.0],
     [4.0, 8.0, 3.0]])
print(x)
x = im2col(x, FH=2, FW=2)
print(x)