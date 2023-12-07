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
sys.path.insert(0, "C:\\Users\\ADMIN\\Desktop\\parallel_project")

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
    processed_img_filename = 'C:/Users/ADMIN/Desktop/parallel-project/uploads/processed_image.png'
    processed_img_path = os.path.join(upload_folder, processed_img_filename)

    # Lưu ảnh đã xử lý
    cv2.imwrite(processed_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))


    

    # Trả về đường dẫn đến ảnh đã xử lý
    return {'processedImage': processed_img_path}

@app.get('/api/getProcessedImage')
async def get_processed_image():
    # Xử lý logic để lấy đường dẫn đến ảnh đã xử lý (hoặc bạn có thể truy cập nó từ một biến đã lưu trữ)

    # Lấy đường dẫn của ảnh đã xử lý
    processed_img_path = "C:/Users/ADMIN/Desktop/parallel-project/uploads/image.png"

    # Kiểm tra xem tệp có tồn tại không
    if not os.path.exists(processed_img_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Trả về tệp ảnh đã xử lý
    return FileResponse(processed_img_path)
