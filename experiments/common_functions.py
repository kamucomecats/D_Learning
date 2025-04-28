import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    softmax_a = a
    y = exp_a/sum_exp_a
    return y

def cross_entropy_error(y, t): #valid only for one-hot case
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    batch_size = y.shape[0]
    y = -np.sum(t * np.log(y)) / batch_size #using broadcast, average loss from all batch is calculated
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))