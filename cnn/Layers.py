import numpy as np
from common import common_functions as cf
from common import im2col as itc
from common import col2im as cti
#variables start from d works in backward
#other variables held in each Layer
    
class ReLULayer:
    def __init__(self):
        self.mask = None
        self.save_cache = True
    
    def forward(self, x):
        out = x.copy()
        out[(x <= 0)] = 0
        if self.save_cache:
            self.mask = (x <= 0)
        ratio = np.count_nonzero(out) / out.size
        print(f"ReLU nonzero ratio: {ratio:.3f}")
        return out
    
    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout
    
class SigmoidLayer:
    def __init__(self):
        self.out = None
        self.save_cache = True
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        if self.save_cache:
            self.out = out
        return out
        
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class AffineLayer: #affine : store W, dW, B, dB by itself
    def __init__(self, params, W_key, b_key):
        self.params = params
        self.W_key = W_key
        self.b_key = b_key
        self.save_cache = True
        self.X = None
        self.dW = None
        self.dB = None
    
    def forward(self, X):
        W = self.params[self.W_key]
        B = self.params[self.b_key]
        if self.save_cache:
            self.X = X # X should renew by forward
        return np.dot(X, W) + B
    
    def backward(self, dY): #same as get X send back dX
        W = self.params[self.W_key]
        self.dW = np.dot(self.X.T, dY)
        self.dB = np.sum(dY, axis=0)
        dX = np.dot(dY, W.T)
        return dX
    
class Softmax_with_lossLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.x = x
        self.t = t
        self.y = cf.softmax(x)
        self.loss = cf.cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    
class DropoutLayer:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.save_cache = True
        self.stride = stride
        self.pad = pad
        self.X = None
    
    def forward(self, X):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = X.shape
        if self.save_cache:
            self.X = X # X should renew by forward
        col = itc.im2col(X, FH, FW, stride=self.stride, pad=self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, C, )
    
    def backward(self, dY): #same as get X send back dX
        W = self.params[self.W_key]
        self.dW = np.dot(self.X.T, dY)
        self.dB = np.sum(dY, axis=0)
        dX = np.dot(dY, W.T)
        return dX
    
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.X_shape = None
        self.X_col = None
        self.argmax = None
    
    def forward(self, X):
        self.X_shape = X.shape
        N, C, H, W = self.X_shape
        OH = (H - self.pool_h) // self.stride + 1
        OW = (W - self.pool_w) // self.stride + 1

        X_col = itc.im2col(X, self.pool_h, self.pool_w, self.stride, self.pad)
        #X_col.shape = (N * OH * OW, C * pool_h * pool_w)
        self.X_col = X_col
        X_col = X_col.reshape(N, C, self.pool_h * self.pool_w, -1)
        #X_col.shape = (N, C, self.pool_h * self.pool_w, OH * OW)
        self.argmax = np.argmax(X_col, axis=2)
        #argmax.shape = ()
        X_col = np.max(X_col, axis=2)
        #X_col.shape = (N, C, OH, OW)
        X_out = X_col.reshape(N, C, OH, OW)
        
        return X_out
    
    def backward(self, dY): #same as get X send back dX
        #dY.shape = (N, C, OH, OW)
        N, C, H, W = self.X_shape
        OH = (H - self.pool_h) // self.stride + 1
        OW = (W - self.pool_w) // self.stride + 1

        dY = dY.reshape(N, C, OH * OW)
        dmax = np.zeros(N, C, OH, OW)
        for n in range(N):
            for c in range(C):
                dmax[n, c, self.argmax[n, c], np.arange(OH * OW)] = dY[n, c]

        dcol = dmax.reshape(N * OH * OW, -1) #受容域ごと
        # 逆変換で元の入力形状に戻す
        dX = cti.col2im(dcol, self.X_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dX
    
x = np.arange(1*1*4*4).reshape(1, 1, 4, 4).astype(np.float32)
print("Original input:")
print(x[0, 0])

col = itc.im2col(x, 2, 2, stride=1, pad=0)
print("im2col result:")
print(col)

x_reconstructed = cti.col2im(col, x.shape, 2, 2, stride=1, pad=0)
print("Reconstructed from col2im:")
print(x_reconstructed[0, 0])
