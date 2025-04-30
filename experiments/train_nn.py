import numpy as np
import mnist_load as ml
import common_functions as cf
import matplotlib.pyplot as plt
import optimizer as op
from batch import batch
from two_layer_net import TwoLayerNet

x_train, t_train = ml.mnist_load()

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 60000
training_rate = 0.01

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

optimizer = op.SGD(lr = training_rate)

for j in range(iters_num):
    x_batch, t_batch = batch(x=x_train, t=t_train, batch_size=100)
    grad = network.gradient(x_batch, t_batch)
#    print(grad)
    

#    for key in grad.keys():
#        print(f"{key} grad mean: {np.mean(np.abs(grad[key]))}")
    
#    for i in (network.params):
#        network.params[i] -= training_rate * grad[i]
    
    optimizer.update(network.params, grad)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
#    if iters_num % 100 == 0:
#        for k, v in grad.items():
#            print(f"{k}: mean={np.mean(np.abs(v)):.6f}, max={np.max(v):.6f}")

    
plt.plot(range(len(train_loss_list)), train_loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Training Loss')
plt.savefig('train_loss.png')

x_test, t_test = ml.mnist_load_test()
acc = network.accuracy(x_test, t_test)
print(f"Test Accuracy: {acc:.4f}")
