import numpy as np
from common import common_functions as cf

#variables start from d works in backward
#other variables held in each Layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
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