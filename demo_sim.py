path_name = "C:\\Users\\ADMIN\\Desktop\\parallel-project"

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, path_name)

from main.utils.puma.puma_ho import puma_ho
from src.APG import APG
from src.func.CCTV import CCTV
from src.func.prox import prox
from main.utils.propagate import propagate


# Load test image
n = 256
img = cv2.resize(cv2.imread( path_name + f'\\data\\simulation\\cameraman.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0, (n, n))


# Sample 
x = np.exp(1j * np.pi * img)

# Physical parameters
pxsize = 5e-3;                  # pixel size (mm)
wavlen = 0.5e-3                 # wavelength (mm)
method = 'Angular Spectrum'     # numerical method
dist   = 5                      # imaging distance (mm)

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

# Calculate padding sizes to avoid circular boundary artifact
kernelsize = dist * wavlen / pxsize / 2
nullpixels = int(np.ceil(kernelsize / pxsize))
x = zeropad(x, nullpixels)



# Define the constraint
global constraint
constraint = 'as'  # 'none': no constraint, 'a': absorption constraint only,
# 's': support constraint only, 'as': absorption + support constraints

# Define the upper bound for the modulus
global absorption
absorption = 1

# Define the support region
global support 
support = np.zeros(x.shape)
support[nullpixels: nullpixels + n, nullpixels: nullpixels + n] = 1





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
    'x2': nullpixels + n,
    'y1': nullpixels + 1,
    'y2': nullpixels + n
}

# Thiết lập hạt giống
np.random.seed(0)
noisevar = 0.01
y = np.abs(A(x)) ** 2

print(y.shape[0], y.shape[1])

y = y * (1 + noisevar * np.random.rand(y.shape[0], y.shape[1])) #Gaussian noise

x_init = AH(np.sqrt(y))
lam = 2e-3
gam = 2
n_iters = 500
n_subiters = 1

# Options
opts = {'verbose': True, 'errfunc': None, 'display': True, 'autosave': False}





def myF(x):
    return F(x, y, A)           # Fidelity function


def mydF(x):
    return dF(x, y, A, AH)      # Gradient of the fidelity function

def myR(x):
    return CCTV(x, lam)         # Regularization function


def myproxR(x, gamma):
    return prox(x, gamma, lam, n_subiters)  # Proximal operator for the regularization function


# Run the algorithm
x_est, J_vals, E_vals, runtimes = APG(x_init, myF, mydF, myR, myproxR, gam, n_iters, opts)



# Display  
# Tạo hình vẽ và cấu hình kích thước
fig = plt.figure(figsize=(12, 4))

# Hiển thị phần amplitudes của x
plt.subplot(1, 3, 1)
plt.imshow(np.abs(x), cmap='gray')
plt.colorbar()
plt.title('Amplitude of the object', fontsize=12)

# Hiển thị phần pha của x
plt.subplot(1, 3, 2)
plt.imshow(np.angle(x), cmap='gray')
plt.colorbar()
plt.title('Phase of the object', fontsize=12)

# Hiển thị y (giả sử đã được định nghĩa từ trước)
plt.subplot(1, 3, 3)
plt.imshow(y, cmap='gray')
plt.colorbar()
plt.title('Intensity measurement', fontsize=12)

# Hiển thị hình vẽ
plt.show()

'''
=============================================================================
    Display results
=============================================================================
'''


# Crop the image
x_crop = x_est[nullpixels:nullpixels + n, nullpixels:nullpixels + n]

print("Running...")

# Hiển thị hình ảnh khôi phục
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
im1 = axs[0].imshow(np.abs(x_crop), cmap='gray')
axs[0].set_title('Retrieved amplitude', fontsize=14)
axs[0].set_xlabel('x-axis')
axs[0].set_ylabel('y-axis')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(np.angle(x_crop), cmap='gray')
axs[1].set_title('Retrieved phase', fontsize=14)
axs[1].set_xlabel('x-axis')
axs[1].set_ylabel('y-axis')
fig.colorbar(im2, ax=axs[1])

plt.show()

print("Running....")

# Hiển thị đường cong hội tụ
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(range(n_iters + 1), E_vals, linewidth=1.5)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Error metric')
axs[0].set_title('Convergence Curve for Error Metric')
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=14)

axs[1].plot(range(n_iters + 1), J_vals, linewidth=1.5)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Objective value')
axs[1].set_title('Convergence Curve for Objective Value')
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()

print("Done!")



