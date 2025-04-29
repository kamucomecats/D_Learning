import numpy as np

def softmax(a):
    if a.ndim == 2:
        c = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / sum_exp_a
    else:
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t): #valid only for one-hot case
    delta = 1e-7    
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    batch_size = y.shape[0]
    y = -np.sum(t * np.log(y + delta)) / batch_size #using broadcast, average loss from all batch is calculated
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))