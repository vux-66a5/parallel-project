import sys 
sys.path.insert(0, 'C:\\Users\\ADMIN\\Desktop\\parallel-project\\src\\func')
# import sys
# import os

# # Lấy đường dẫn của file hiện tại
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Xác định đường dẫn tương đối đến thư mục chứa module của bạn
# module_dir = os.path.join(current_dir, '..')

# # Thêm đường dẫn tương đối vào sys.path để Python có thể tìm thấy module của bạn
# sys.path.insert(0, module_dir)

from constraints.indicator import indicator
from normTV import normTV

def CCTV(x, lam):
    '''
    =============================================================================
    The constrained complex total variation (CCTV) function.
    
    Input:          - x   : The complex-valued 2D transmittance of the sample.
                    - lam : The regularization parameter for the total variation function.
    
    Output:         - val : Value of the CCTV function.
    =============================================================================
    '''
    val = normTV(x, lam) + indicator(x)

    return val

