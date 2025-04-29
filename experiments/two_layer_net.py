import numpy as np
from Layers import *
import common_functions as cf
from collections import OrderedDict
import grad as gr

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=1.0):
        self.params={}
        # 例: He初期化で W1 を初期化
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params, 'W1', 'b1')
        self.layers['Relu1']   = ReLULayer()
        self.layers['Affine2'] = AffineLayer(self.params, 'W2', 'b2')
        self.lastLayer = Softmax_with_lossLayer()
        
    def predict(self, x, save_cache=False): #assume x.shape = [1, 784]
        for layer in self.layers.values():
            if hasattr(layer, 'save_cache'):
                layer.save_cache = save_cache
        out = x
        for layer in self.layers.values():
            out = layer.forward(out)#assume y.shape = [1, 10] (one-hot output)
        return out
    
    def loss(self, x, t, save_cache=True):
        y = self.predict(x, save_cache)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        size = x.shape[0]
        y = self.predict(x)
        arg_y = np.argmax(y, axis=1) #conpress [batch_size, 10] into [batch_size] (array like 5, 2, 4, ...)
        arg_t = np.argmax(t, axis=1) #conpress [batch_size, 10] into [batch_size] (array like 5, 7, 4, ...)
        rate = (float)(np.sum(arg_y==arg_t)) / size
        return rate
        
    def numerical_gradient(self, x, t):
        grads = {}
        h = 1e-4

        for key in ('W1', 'b1', 'W2', 'b2'):
            param = self.params[key]
            grad = np.zeros_like(param)

            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                tmp_val = param[idx]

                # (1) パラメータを +h だけずらす
                param[idx] = tmp_val + h
                loss1 = self.loss_for_numerical_gradient(x, t)

                # (2) パラメータを -h だけずらす
                param[idx] = tmp_val - h
                loss2 = self.loss_for_numerical_gradient(x, t)

                # (3) 勾配計算
                grad[idx] = (loss1 - loss2) / (2 * h)

                # (4) 元に戻す
                param[idx] = tmp_val

                it.iternext()

            grads[key] = grad

        return grads

    
    def loss_for_numerical_gradient(self, x, t):
        # save_cache=False で予測だけする
        y = self.predict(x, save_cache=False)
        y = cf.softmax(y)
        return cf.cross_entropy_error(y, t)

    
    def gradient(self, x, t):
        #forward
        
        self.loss(x, t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].dB
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].dB
        
        return grads