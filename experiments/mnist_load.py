import os
import numpy as np

filename = 'mnist_local.npz'

def to_one_hot(t, num_classes=10):
    if t.ndim == 1:
        t_one_hot = np.zeros((t.size, num_classes))
        t_one_hot[np.arange(t.size), t] = 1
    elif t.ndim == 2 and t.shape[1] == 1:
        t = t.flatten()
        t_one_hot = np.zeros((t.size, num_classes))
        t_one_hot[np.arange(t.size), t] = 1
    else:
        raise ValueError("Invalid shape for label array")
    return t_one_hot

def mnist_load():
    if os.path.exists(filename):
        with np.load(filename) as data:
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
        print("✅ ローカルから読み込みました")
    else:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        np.savez_compressed(filename,
                            x_train=x_train, y_train=y_train,
                            x_test=x_test, y_test=y_test)
        print("⬇️ ダウンロード＆保存しました")
    y_train = to_one_hot(y_train)
    
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    return x_train, y_train

def mnist_load_test():
    if os.path.exists(filename):
        with np.load(filename) as data:
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
        print("✅ ローカルから読み込みました")
    else:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        np.savez_compressed(filename,
                            x_train=x_train, y_train=y_train,
                            x_test=x_test, y_test=y_test)
        print("⬇️ ダウンロード＆保存しました")
    y_test = to_one_hot(y_test)
    
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    return x_test, y_test