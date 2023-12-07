import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def propagate(w_i, dist, pxsize, wavlen, method):
    # Calculate the free-space propagation of a complex wavefield.

    # Input:
    # - w_i    : Input complex wavefield
    # - dist   : Propagation distance
    # - pxsize : Pixel size
    # - wavlen : Wavelength
    # - method : Numerical method ('Fresnel' or 'Angular Spectrum')

    # Output:
    # - w_o    : Output wavefield after propagation

    nx, ny = np.array(w_i).shape  # size of the wavefront

    # sampling in the spatial frequency domain
    kx = np.pi / pxsize * np.linspace(-1, 1 - 2 / ny, ny)
    ky = np.pi / pxsize * np.linspace(-1, 1 - 2 / nx, nx)
    KX, KY = np.meshgrid(kx, ky)

    k = 2 * np.pi / wavlen  # wave number

    # remove evanescent orders 
    ind = (KX ** 2 + KY ** 2 >= k ** 2)
    KX[ind] = 0
    KY[ind] = 0

    inputFT = fftshift(fft2(w_i))

    if method == 'Fresnel':
        H = np.exp(1j * k * dist) * np.exp(-1j * dist * (KX ** 2 + KY ** 2) / 2 / k)
    elif method == 'Angular Spectrum':
        H = np.exp(1j * dist * np.sqrt(k ** 2 - KX ** 2 - KY ** 2))
    else:
        raise ValueError('Wrong parameter for [method]: must be <Angular Spectrum> or <Fresnel>')

    w_o = ifft2(fftshift(inputFT * H))

    return w_o

'''
x = [
    [-0.1993 - 0.1069j, 0.2086 + 0.1119j, -0.2301 - 0.1053j, 0.2621 + 0.0863j, -0.3008 - 0.0524j],
    [0.1591 + 0.1649j, -0.1190 - 0.1023j, 0.0835 + 0.0387j, -0.0455 + 0.0183j, 0.0008 - 0.0598j],
    [-0.1354 - 0.2339j, 0.0375 + 0.1128j, 0.0613 - 0.0018j, -0.1722 - 0.0829j, 0.2982 + 0.1216j],
    [0.1389 + 0.3152j, 0.0302 - 0.1537j, -0.2022 + 0.0137j, 0.3898 + 0.0792j, -0.5924 - 0.0952j],
    [-0.1831 - 0.4035j, -0.0707 + 0.2313j, 0.3242 - 0.0915j, -0.5882 + 0.0197j, 0.8546 - 0.0536j]
]

print(propagate(x, 8.5, 0.0059, 0.00066, 'Angular Spectrum'))'''
