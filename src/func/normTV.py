# import sys 
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
import numpy as np



def normTV(x, lam):
    '''
    ===================================================================
    Calculate the total variation (TV) seminorm for an input image.
    -------------------------------------------------------------------
    Input:      - x         : 2D image.
                - lam       : Regularization weight.
    Output:     - n         : The function value.
    -------------------------------------------------------------------
    ===================================================================
    '''

    def norm1(x):
        return np.sum(np.abs(x))

    g = D(x)
    n = lam * norm1(g[:, :, 0]) + lam * norm1(g[:, :, 1])

    
    return n



