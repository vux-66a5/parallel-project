path_name = "C:\\Users\\vuxxw\\PycharmProjects\\Group16\\parallel-project"
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import cv2
import numpy as np
import os
from io import BytesIO
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


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


app = FastAPI()
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Điều chỉnh điều này tùy theo yêu cầu của bạn, * để cho phép tất cả các origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/api/processImage')
async def process_image(file: UploadFile = File(...)):
    # Kiểm tra file upload
    if not file:
        raise HTTPException(status_code=400, detail='Vui lòng chọn một file.')

    # Đọc file ảnh
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    # Xử lý ảnh
    # ... (thay thế bằng code xử lý ảnh của bạn)






    processed_img = img

    # Thư mục lưu trữ ảnh đã xử lý
    upload_folder = 'uploads'  # Tên thư mục bạn muốn lưu trữ ảnh
    os.makedirs(upload_folder, exist_ok=True)  # Tạo thư mục nếu nó chưa tồn tại

    # Tên file và đường dẫn để lưu ảnh đã xử lý
    processed_img_filename = 'C:/Users/vuxxw/PycharmProjects/Group16/parallel-project/uploads/processed_image.png'
    processed_img_path = os.path.join(upload_folder, processed_img_filename)

    # Lưu ảnh đã xử lý
    cv2.imwrite(processed_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))



    # Load test image
    n = 256
    img = cv2.resize(cv2.imread('C:/Users/vuxxw/PycharmProjects/Group16/parallel-project/uploads/processed_image.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0, (n, n))


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

    def F(x, y, A):
        def norm2(x):
            n = np.linalg.norm(x)
            return n 
        
        v = 1/2 * norm2(np.abs(A(x)) - np.sqrt(y))**2


    def dF(x, y, A, AH):
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

    plt.ioff()
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
    plt.savefig('C:/Users/vuxxw/PycharmProjects/Group16/parallel-project/uploads/image.png', bbox_inches='tight')
    plt.ion()


    plt.show()


    # Trả về đường dẫn đến ảnh đã xử lý
    return {'processedImage': processed_img_path}

@app.get('/api/getProcessedImage')
async def get_processed_image():
    # Xử lý logic để lấy đường dẫn đến ảnh đã xử lý (hoặc bạn có thể truy cập nó từ một biến đã lưu trữ)

    # Lấy đường dẫn của ảnh đã xử lý
    processed_img_path = "C:/Users/vuxxw/PycharmProjects/Group16/parallel-project/uploads/image.png"

    # Kiểm tra xem tệp có tồn tại không
    if not os.path.exists(processed_img_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Trả về tệp ảnh đã xử lý
    return FileResponse(processed_img_path)