import numpy as np

def DT(w):
    '''
    ==============================================================
    Calculate the transpose of the gradient operator (D).
    --------------------------------------------------------------
    Input:         - w  : 3D array.
    Output         - u  : 2D array.
    --------------------------------------------------------------
    ==============================================================
    '''
    n1, n2, _ = w.shape

    shift = np.roll(w[:, :, 0], shift=(1, 0), axis=(0, 1))
    u1 = w[:, :, 0] - shift
    u1[0, :] = w[0, :, 0]
    u1[n1 - 1, :] = -shift[n1 - 1, :]

    shift = np.roll(w[:, :, 1], shift=(0, 1), axis=(0, 1))
    u2 = w[:, :, 1] - shift
    u2[:, 0] = w[:, 0, 1]
    u2[:, n2 - 1] = -shift[:, n2 - 1]

    u = u1 + u2

    return u


