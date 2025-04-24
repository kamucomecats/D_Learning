import numpy as np

def batch(x, t, batch_size=100):
    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch, t_batch = x[batch_mask], t[batch_mask]
    return x_batch, t_batch