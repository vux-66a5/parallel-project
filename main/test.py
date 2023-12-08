path_name = "C:\\Users\\ADMIN\\Desktop\\parallel-project"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import sys
sys.path.insert(0, path_name)

from main.utils.puma.puma_ho import puma_ho
from src.APG import APG
from src.func.CCTV import CCTV
from src.func.normTV import normTV
from src.func.prox import prox
from main.utils.propagate import propagate
from src.func.constraints.indicator import indicator 
from flask import Flask, jsonify
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/api/processImage', methods=['POST'])
def process_image():
    # Tải ảnh từ URL
    image_url = 'http://localhost:3000/api/processImage'  # Thay đổi URL của bạn ở đây
    response = requests.get(image_url)
    if response.status_code == 200:
        try:
            # Đọc dữ liệu ảnh từ response.content với imageio
            image = imageio.imread(BytesIO(response.content))
            # Bây giờ 'image' chứa dữ liệu ảnh gốc
            # Bạn có thể sử dụng 'image' cho các xử lý tiếp theo mà không cần chuyển đổi sang grayscale
        except Exception as e:
            return jsonify({'error': f'Failed to process image: {str(e)}'})
    else:
        return jsonify({'error': 'Failed to fetch image from URL'})

   
    # Load the background and object images
    group_num = 1
    bg_path = path_name + f'\\data\\experiment\\E{group_num}\\bg.bmp'
    obj_path = path_name + f'\\data\\experiment\\E{group_num}\\obj.bmp'

    img_bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)  # Read and convert to grayscale
    img_bg = img_bg.astype(np.float64) / 255.0

    img_obj = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)  # Read and convert to grayscale
    img_obj = img_obj.astype(np.float64) / 255.0


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
    x_init = AH(np.sqrt(y))  # Initial guess
    lam = 1e-2               # Regularization parameter
    gam = 2                  # Step size
    n_iters = 500            # Number of iterations (main loop)
    n_subiters = 7           # Number of iterations (denoising)

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

    print("Run Complete!")

    # Visualize the reconstructed image
    plt.figure(figsize=(12, 6))

    # Plot the retrieved amplitude on the left subplot
    plt.subplot(1, 2, 1)
    plt.imshow(amp_est, cmap='gray', vmin=0, vmax=np.max(amp_est))  # You can choose the colormap you prefer
    plt.colorbar()
    plt.title('Retrieved amplitude', fontsize=14)
    plt.axis('off')

    # Plot the retrieved phase on the right subplot
    plt.subplot(1, 2, 2)
    plt.imshow(pha_est, cmap='gray', vmin=np.min(pha_est), vmax=np.max(pha_est))  # You can choose the colormap you prefer
    plt.colorbar()
    plt.title('Retrieved phase', fontsize=14)
    plt.axis('off')

    # Adjust the position of the figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Display the figure
    plt.show()
    processed_img = plt.savefig('processed_img.png', format='png', dpi=100)
    # Sau khi có ảnh đã xử lý, chuyển đổi ảnh thành dạng base64
    _, img_encoded = cv2.imencode('.png', processed_img)
    processed_img_bytes = img_encoded.tobytes()
    processed_image_base64 = f'data:image/png;base64,{base64.b64encode(processed_img_bytes).decode()}'

    return jsonify({'processedImage': processed_image_base64})

if __name__ == '__main__':
    app.run(debug=True)
