path_name = "C:\\Users\\ADMIN\\Desktop\\parallel-project"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft2, ifft2
import matplotlib.pyplot as plt
import cupy as cp

import sys
sys.path.insert(0, path_name)

from main.utils.puma.puma_ho import puma_ho
from src.APG import APG
from src.func.CCTV import CCTV
from src.func.prox import prox
from main.utils.propagate import propagate

# Load the background and object images
group_num = 1
bg_path = path_name + f'\\data\\experiment\\E{group_num}\\bg.bmp'
obj_path = path_name + f'\\data\\experiment\\E{group_num}\\obj.bmp'

img_bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)  # Read and convert to grayscale
img_bg = img_bg.astype(np.float64) / 255.0

img_obj = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)  # Read and convert to grayscale
img_obj = img_obj.astype(np.float64) / 255.0

'''
# Load parameters from a .mat file
params_path = 'C:/Users/This Pc/Desktop/TTSS-cuoiky-master/data/experiment/E1/params.mat'
data = loadmat(params_path)
params = data['params']
pxsize = params[0][0]['pxsize'][0][0]
wavlen = params[0][0]['wavlen'][0][0]
dist = params[0][0]['dist'][0][0]
method = 'Angular Spectrum'
'''

pxsize = 0.0059
wavlen = 0.00066
dist = 8.5000
method = 'Angular Spectrum'


# Normalize the hologram
y = img_obj / np.mean(img_bg)

# Select area of interest for reconstruction
# Hiển thị hình ảnh và cho phép người dùng chọn vùng quan tâm
rect_tuple = cv2.selectROI(img_obj)

rect = list(rect_tuple)

# Cắt vùng quan tâm từ hình ảnh gốc 
temp = img_obj[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]

if temp.shape[0] % 2 == 1:
    rect[3] -= 1

if temp.shape[1] % 2 == 1:
    rect[2] -= 1

# Đóng cửa sổ hiển thị hình ảnh
cv2.destroyAllWindows()

# Crop 'y' using the adjusted rectangle
y = y[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]

# Get the dimensions of the cropped 'y'
n1, n2 = y.shape

# Calculate padding sizes to avoid circular boundary artifact
kernelsize = dist * wavlen / pxsize / 2
nullpixels = int(np.ceil(kernelsize / pxsize))

# Pre-calculate the transfer functions for diffraction modeling
def transfunc(nx, ny, dist, pxsize, wavlen, method):
    kx = np.pi / pxsize * np.linspace(-1, 1 - 2 / nx, nx)
    ky = np.pi / pxsize * np.linspace(-1, 1 - 2 / ny, ny)
    KX, KY = np.meshgrid(kx, ky)

    k = 2 * np.pi / wavlen

    ind = (KX ** 2 + KY ** 2) >= k ** 2
    KX[ind] = 0
    KY[ind] = 0

    if method.lower() == 'fresnel':
        H = np.exp(1j * k * dist) * np.exp(-1j * dist * (KX ** 2 + KY ** 2) / (2 * k))
    elif method.lower() == 'angular spectrum':
        H = np.exp(1j * dist * np.sqrt(k ** 2 - KX ** 2 - KY ** 2))
    else:
        raise ValueError("Invalid method. Should be 'Angular Spectrum' or 'Fresnel'.")

    return H

def Df(x):
    """
    Calculate the 2D gradient (finite difference) of an input image.
    
    Parameters:
    - x: The input 2D image.
    
    Returns:
    - w: The gradient (3D array).
    """
    # Calculate differences along rows and columns
    diff_rows = x[1:, :] - x[:-1, :]
    diff_cols = x[:, 1:] - x[:, :-1]
    
    # Extend differences to match the size of the input image
    diff_rows = np.vstack((diff_rows, np.zeros_like(x[-1, :])))
    diff_cols = np.hstack((diff_cols, np.zeros_like(x[:, -1:]) ))
    
    # Stack differences along the third dimension to get the gradient
    w = np.stack((diff_rows, diff_cols), axis=2)
    
    return w

def DTf(w):
    """
    Calculate the transpose of the gradient operator.
    
    Parameters:
    - w: 3D array.
    
    Returns:
    - u: 2D array.
    """
    # Extract components from the 3D array
    w1 = w[:, :, 0]
    w2 = w[:, :, 1]
    
    # Calculate the transpose
    u1 = w1 - np.roll(w1, 1, axis=0)
    u1[0, :] = w1[0, :]
    u1[-1, :] = -w1[-2, :]
    
    u2 = w2 - np.roll(w2, 1, axis=1)
    u2[:, 0] = w2[:, 0]
    u2[:, -1] = -w2[:, -2]
    
    # Sum components to get the result
    u = u1 + u2
    
    return u


H_f = fftshift(transfunc(n2 + nullpixels * 2, n1 + nullpixels * 2, dist, pxsize, wavlen, method))  # Forward propagation
H_b = fftshift(transfunc(n2 + nullpixels * 2, n1 + nullpixels * 2, -dist, pxsize, wavlen, method))  # Backward propagation


gpu = True

# Define the constraint
global constraint
constraint = 'a'  # 'none': no constraint, 'a': absorption constraint only,
# 's': support constraint only, 'as': absorption + support constraints

# Define the upper bound for the modulus
global absorption
absorption = 1.1

# Define the support region
global support 
support = np.zeros((n1 + nullpixels * 2, n2 + nullpixels * 2))
support[nullpixels: nullpixels + n1, nullpixels: nullpixels + n2] = 1

# Forward model
# Forward Propagation
def Q(x):
    return propagate(x, dist, pxsize, wavlen, method)

# Hermitian of Q: backward propagation
def QH(x):
    return propagate(x, -dist, pxsize, wavlen, method)

# Image cropping operation (to model the finite size of the sensor area)
def imgcrop(x, cropsize):
    """
    Crop the central part of the image.

    Input:
    - x: Original image.
    - cropsize: Cropping pixels.
    """
    '''
    crop_start = (np.array(x.shape) - cropsize) // 2
    crop_end = crop_start + cropsize
    u = x[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]
    '''

    u = x[cropsize:-cropsize, cropsize:-cropsize]

    return u

def C(x):
    return imgcrop(x, nullpixels)

# Overall sampling operation
def A(x):
    return C(Q(x))

def zeropad(x, padsize):
    # Zero-pad the image.
    # Input:    - x        : Original image.
    #           - padsize  : Padding pixel number along each dimension.
    # Output:   - u        : Zero-padded image.
    u = np.pad(x, padsize, mode='constant', constant_values=0)
    return u

# Transpose of C: zero-padding operation
def CT(x):
    return zeropad(x, nullpixels)

# Hermitian of A
def AH(x):
    return QH(CT(x))


'''
====================================================================================
    Auxiliary functions
====================================================================================
'''

def F(x, y, A):
    '''
    % =========================================================================
    % Data-fidelity function.
    % -------------------------------------------------------------------------
    % Input:    - x   : The complex-valued transmittance of the sample.
    %           - y   : Intensity image.
    %           - A   : The sampling operator.
    % Output:   - v   : Value of the fidelity function.
    % =========================================================================
    '''
    def norm2(x):
        n = np.linalg.norm(x)
        return n 
    
    v = 1/2 * norm2(np.abs(A(x)) - np.sqrt(y))**2


def dF(x, y, A, AH):
    """
    Gradient of the data-fidelity function.

    Input:
    - x: The complex-valued transmittance of the sample.
    - y: Intensity image.
    - A: The sampling operator.
    - AH: Hermitian of A.

    Output:
    - g: Wirtinger gradient.
    """
    u = A(x)
    u = (np.abs(u) - np.sqrt(y)) * np.exp(1j * np.angle(u))
    g = 1 / 2 * AH(u)

    return g



region = {
    'x1': nullpixels + 1,
    'x2': nullpixels + n1,
    'y1': nullpixels + 1,
    'y2': nullpixels + n2
}

# Algorithm settings
x_est = AH(np.sqrt(y))  # Initial guess
lam = 1e-2               # Regularization parameter
gam = 2                  # Step size
n_iters = 500            # Number of iterations (main loop)
n_subiters = 7           # Number of iterations (denoising)


# Auxilary variables 
z_est = x_est
v_est = np.zeros((x_est.shape[0], x_est.shape[1], 2))
w_est = np.zeros((x_est.shape[0], x_est.shape[1], 2))

if gpu:
    cp.cuda.Device(0).use()  # Chọn thiết bị GPU, 0 là số thứ tự của GPU (thay đổi nếu có nhiều GPU)
    x_est = cp.asarray(x_est)
    y = cp.asarray(y)
    H_f = cp.asarray(H_f)
    H_b = cp.asarray(H_b)
    support = cp.asarray(support)
    z_est = cp.asarray(z_est)
    v_est = cp.asarray(v_est)
    w_est = cp.asarray(w_est)


def projf(x, constraint='none', absorption=None, support=None):
    """
    Parameters:
    - x: NumPy array, input array.
    - constraint: str, constraint type ('none', 'a', 's', 'as').
    - absorption: NumPy array, absorption array (only used for 'a' and 'as' constraints).
    - support: NumPy array, support array (only used for 's' and 'as' constraints).

    Returns:
    - result: NumPy array, projected array.
    """
    if constraint.lower() == 'none':
        result = x
    elif constraint.lower() == 'a':
        result = np.minimum(np.abs(x), np.abs(absorption)) * np.exp(1j * np.angle(x))
    elif constraint.lower() == 's':
        result = x * support
    elif constraint.lower() == 'as':
        result = np.minimum(np.abs(x), np.abs(absorption)) * np.exp(1j * np.angle(x)) * support
    else:
        raise ValueError("Invalid constraint. Should be 'a' (absorption), 's' (support), 'as' (both), or 'none'.")
    
    return result

'''timer = cp.cuda.Event()
timer.record()'''



for iter in range(1, n_iters + 1):
    # In trạng thái
    print(f'iter: {iter} / {n_iters}')
    
    # Gradient update
    u = A(cp.asnumpy(z_est))
    u = cp.asarray(u)
    u = 1/2 * cp.asarray(AH(cp.asnumpy((cp.abs(u) - cp.sqrt(y)) * cp.exp(1j * cp.angle(u)))))
    u = z_est - gam * u
    
    # Proximal update
    v_est[:] = 0
    w_est[:] = 0
    
    for subiter in range(1, n_subiters + 1):
        w_next = v_est + 1/8/gam*Df(projf(u - gam*DTf(v_est)))
        w_next = cp.minimum(cp.abs(w_next), lam) * cp.exp(1j * cp.angle(w_next))
        v_est = w_next + subiter / (subiter + 3) * (w_next - w_est)
        w_est = w_next
    
    x_next = projf(u - gam * DTf(w_est))
    
    # Nesterov extrapolation
    z_est = x_next + (iter / (iter + 3)) * (x_next - x_est)
    x_est = x_next

'''# Đo thời gian
timer.synchronize()
print(f'Time elapsed: {timer.time_till()} seconds')'''

# Chờ cho GPU (nếu có)
if gpu:
    cp.cuda.Stream.null.synchronize()
    
    # Thu thập dữ liệu từ GPU
    x_est = cp.asnumpy(x_est)
    y = cp.asnumpy(y)
    H_f = cp.asnumpy(H_f)
    H_b = cp.asnumpy(H_b)
    support = cp.asnumpy(support)



def myF(x):
    return F(x, y, A)           # Fidelity function


def mydF(x):
    return dF(x, y, A, AH)      # Gradient of the fidelity function

def myR(x):
    return CCTV(x, lam)         # Regularization function


def myproxR(x, gamma):
    return prox(x, gamma, lam, n_subiters)  # Proximal operator for the regularization function



'''
=============================================================================
    Display results
=============================================================================
'''



# Crop the image
x_crop = x_est[nullpixels:nullpixels + n1, nullpixels:nullpixels + n2]

# Compute the amplitude
amp_est = np.abs(x_crop)

print('Running...!')

# Compute the phase 
pha_est = puma_ho(np.angle(x_crop), 1)[0]

# # Khởi tạo hình với 2 subplot
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# # Hiển thị ảnh amplitudes
# axs[0].imshow(amp_est, cmap='gray', vmin=0, vmax=amp_est.max())
# axs[0].set_title('Retrieved amplitude', fontsize=14)
# axs[0].set_colorbar()
# axs[0].axis('off')

# # Hiển thị ảnh phases
# axs[1].imshow(pha_est, cmap='gray', vmin=pha_est.min(), vmax=pha_est.max())
# axs[1].set_title('Retrieved phase', fontsize=14)
# axs[1].set_colorbar()
# axs[1].axis('off')

# # Đặt kích thước và vị trí của hình
# fig.subplots_adjust(left=0.2, right=0.8)

# # Hiển thị hình
# plt.show()
# Khởi tạo hình với 2 subplot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Hiển thị ảnh amplitudes
img1 = axs[0].imshow(amp_est, cmap='gray', vmin=0, vmax=amp_est.max())
axs[0].set_title('Retrieved amplitude', fontsize=14)
# Thêm colorbar cho subplot 0
cbar1 = plt.colorbar(img1, ax=axs[0])
axs[0].axis('off')

# Hiển thị ảnh phases
img2 = axs[1].imshow(pha_est, cmap='gray', vmin=pha_est.min(), vmax=pha_est.max())
axs[1].set_title('Retrieved phase', fontsize=14)
# Thêm colorbar cho subplot 1
cbar2 = plt.colorbar(img2, ax=axs[1])
axs[1].axis('off')

# Đặt kích thước và vị trí của hình
fig.subplots_adjust(left=0.2, right=0.8)

# Hiển thị hình
plt.show()
