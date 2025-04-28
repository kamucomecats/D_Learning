import numpy as np
import mnist_load as ml
import common_functions as cf
from batch import batch
from two_layer_net import TwoLayerNet

x_train, t_train = ml.mnist_load()

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
training_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

for _ in range(iters_num):
    x_batch, t_batch = batch(x=x_train, t=t_train, batch_size=100)
    grad = network.numerical_gradient(x_batch, t_batch)

    
    for i in (['W1', 'W2', 'b1', 'b2']):
        network.params[i] -= training_rate * grad[i]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)