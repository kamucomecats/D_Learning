import numpy as np

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
            
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.momentum = momentum
        self.lr = lr
        self.v = None
        
    def update(self, params, grads):
        if self.v == None: #first time use SGD
            self.v = {}
            for key, val in params.items():
                self.v[key] = -self.lr * grads[key]
                params[key] += self.v[key]
            return
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key]-self.lr*grads[key]
            params[key] += self.v[key]

