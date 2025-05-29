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

def _load_mnist_data():
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
    return x_train, y_train, x_test, y_test

def mnist_load(flatten=True, one_hot=True):
    x_train, y_train, x_test, y_test = _load_mnist_data()

    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    if one_hot:
        y_train = to_one_hot(y_train)
        y_test = to_one_hot(y_test)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        y_test = y_test.reshape(y_test.shape[0], -1)

    return x_train, y_train, x_test, y_test
