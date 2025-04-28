import numpy as np

def gradient(f, x): #初回の仕様は通常の微分で実装、
    x = x.astype(np.float64)
    grad=np.zeros_like(x)
    h=1e-4
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        print(i)
        tmp=x[i]
        x[i] = tmp + h
        fxh1=f(x)
        x[i] = tmp - h
        fxh2=f(x)
        x[i] = tmp
        grad[i] = (fxh1 - fxh2) / (2*h)
        
        it.iternext()
    return grad
