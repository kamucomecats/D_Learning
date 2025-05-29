import numpy as np
from cnn import Layers
from common import common_functions as cf
from collections import OrderedDict
from common import grad as gr

#W is 4-dim constantly

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, \
        'filter_size':5, 'pad':0, 'stride':1}, hidden_size=100, output_size=10, \
            weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) /\
        filter_stride + 1
        pool_output_size = int(filter_num * (hidden_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], \
                                                                filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], \
                                            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = ReluLayer()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
    
    def predict(self, x, save_cache=False, train_flg=True): #assume x.shape = [1, 784]
        for layer in self.layers.values():
            if hasattr(layer, 'save_cache'):
                layer.save_cache = save_cache
            if hasattr(layer, 'Train_flg'):
                layer.train_flg = train_flg
        out = x
        for layer in self.layers.values():
            out = layer.forward(out)#assume y.shape = [1, 10] (one-hot output)
        return out
    
    def loss(self, x, t, save_cache=True, train_flg=False): #cache is off only when predict called from loss
        y = self.predict(x, save_cache, train_flg)
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
    