import numpy as np
import common_functions as cf
import grad as gr

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x): #assume x.shape = [1, 784]
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        A1 = np.dot(x, W1) + b1
        Z = cf.sigmoid(A1)
        A2 = np.dot(Z, W2) + b2
        y = cf.sigmoid(A2) #assume y.shape = [1, 10] (one-hot output)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cf.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        size = x.shape[0]
        y = self.predict(x)
        arg_y = np.argmax(y, axis=1) #conpress [batch_size, 10] into [batch_size] (array like 5, 2, 4, ...)
        arg_t = np.argmax(t, axis=1) #conpress [batch_size, 10] into [batch_size] (array like 5, 7, 4, ...)
        rate = (float)(np.sum(arg_y==arg_t)) / size
        return rate
    
    def numerical_gradient(self, x, t):
        grads = {}
        loss_W = lambda W : self.loss(x, t)
        grads['W1'] = gr.gradient(loss_W, self.params['W1'])
        grads['W2'] = gr.gradient(loss_W, self.params['W2'])
        grads['b1'] = gr.gradient(loss_W, self.params['b1'])
        grads['b2'] = gr.gradient(loss_W, self.params['b2'])
        return grads