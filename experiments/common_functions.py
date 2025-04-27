import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    softmax_a = a
    y = exp_a/sum_exp_a
    return y

def cross_entropy_error(x, t): #valid only for one-hot case
    d=1e-4
    y=-np.sum(t*np.log(x+d))
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))