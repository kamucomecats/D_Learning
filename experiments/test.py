import numpy as np
import mnist_load as ml
import batch as bt

x_train, t_train = ml.mnist_load()

print(x_train.shape)

x_batch, t_batch = bt.batch(x_train, t_train)

print(x_batch.shape)