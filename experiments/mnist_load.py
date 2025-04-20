import os
import numpy as np

filename = 'mnist_local.npz'

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
    return x_train, y_train