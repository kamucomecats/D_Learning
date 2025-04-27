import numpy as np

def gradient(f, x): #初回の仕様は通常の微分で実装、
    grad=np.zeros_like(x, dtype=np.float64) #多次元誤差逆の順
    h=1e-4
    for i in range(x.size):
        tmp=x[i]
        x[i] = tmp + h
        fxh1=f(x)
        x[i] = tmp - h
        fxh2=f(x)
        x[i] = tmp
        grad[i] = (fxh1 - fxh2) / (2*h)
        print("loop")
    return grad

def function_1(x):
    return x[0]**2+x[1]**2


x=np.array([3.0, 4.0])
print(function_1(x))
print(gradient(function_1, x))