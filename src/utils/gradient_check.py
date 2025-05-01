import numpy as np
import mnist_load as ml
from two_layer_net import TwoLayerNet

x_train, t_train = ml.mnist_load()

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    num = grad_numerical[key]
    bp = grad_backprop[key]
    diff = np.average(np.abs(bp - num))
    rel_error = np.sum(np.abs(bp - num)) / (np.sum(np.abs(bp)) + np.sum(np.abs(num)))
    print(f"{key} : absolute diff = {diff}, relative error = {rel_error}")
