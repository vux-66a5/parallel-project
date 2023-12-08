path_name = "C:\\Users\\ADMIN\\Desktop\\parallel-project"

import sys 
sys.path.insert(0,  path_name + f'\\src\\func')

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



