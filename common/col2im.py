import numpy as np

def col2im(col, input_shape, FH, FW, stride=1, pad=0):
    N, C, H, W = input_shape
    OH = (H + 2 * pad - FH) // stride + 1
    OW = (W + 2 * pad - FW) // stride + 1

    col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
