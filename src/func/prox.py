import numpy as np
# import sys 
# sys.path.insert(0, '"C:\\Users\\This Pc\\Desktop\\parallel-project\\src\\func"')
# sys.path.insert(0, '"C:\\Users\\This Pc\\Desktop\\parallel-project\\src\\func"')
import sys
import os

# Lấy đường dẫn của file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Xác định đường dẫn tương đối đến thư mục chứa module của bạn
module_dir = os.path.join(current_dir, '..')

# Thêm đường dẫn tương đối vào sys.path để Python có thể tìm thấy module của bạn
sys.path.insert(0, module_dir)

from D import D
from DT import DT
from constraints.proj import proj

def prox(v, gam, lam, n_iters):
    '''
    ==========================================================
    Calculate the proximal operator for the CCTV function
    ----------------------------------------------------------
    Input:      - v         : 2D image.
                - gam       : the step size.
                - lam       : the regularization weight.
                - n_iters   : the iteration number.
    Output:     - x         : 2D array.
    ==========================================================
    '''

    n1, n2 = np.shape(v)
    w = np.zeros((n1, n2, 2))
    w_prev = np.zeros((n1, n2, 2))
    z = np.zeros((n1, n2, 2))

    for t in range(1, n_iters+1):
        w = z + 1/8/gam * D(proj(v-gam*DT(z)))
        w = np.minimum(np.abs(w), lam) * np.exp(1j * np.angle(w))

        z = w + t/(t+3) * (w-w_prev)

        w_prev = w

    x = proj(v - gam*DT(w))
    return x

