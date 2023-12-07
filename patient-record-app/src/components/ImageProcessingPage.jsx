import React, { useState } from 'react';
import axios from 'axios';
import './style.css';

const ImageProcessingPage = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);

    // Xử lý chọn file
    const handleFileChange = (event) => {
        // Lưu file đã chọn vào state
        setSelectedFile(event.target.files[0]);
    };

    // Xử lý khi bắt đầu xử lý ảnh
    const handleImageProcessing = async () => {
        if (selectedFile) {
            try {
                const formData = new FormData();
                formData.append('file', selectedFile);

                // Gửi file và yêu cầu xử lý lên server
                const response = await axios.post('http://127.0.0.1:8000/api/processImage', formData);

                // Kiểm tra phản hồi từ server
                if (response.data.error) {
                    console.error('Error:', response.data.error);
                    return;
                }

                // Lấy đường dẫn đến ảnh đã xử lý từ phản hồi
                const processedImagePath = response.data.processedImage;

                // Cập nhật state với ảnh đã xử lý
                setProcessedImage(processedImagePath); // Cập nhật trạng thái ảnh đã xử lý

                // Thêm code để cập nhật đường dẫn của ảnh sau khi xử lý vào src của thẻ img
                // Đảm bảo rằng `processedImagePath` chứa URL chính xác của ảnh đã xử lý trên server
            } catch (error) {
                console.error('Error processing image:', error);
            }
        } else {
            console.log('Vui lòng chọn một file trước khi bắt đầu xử lý.');
        }
    };


    return (
        <div className="background">
            <div className="img-container">
                <h2>Trang xử lý ảnh</h2>
                <input type="file" onChange={handleFileChange} />
                <button className="button" onClick={handleImageProcessing}>Bắt đầu</button>
                {/* Hiển thị ảnh trước và sau khi xử lý */}
                {selectedFile && (
                    <div className="image-container">
                        <div className="image-wrapper">
                            <h3>Ảnh trước khi xử lý</h3>
                            <img src={URL.createObjectURL(selectedFile)} alt="Before processing" />
                        </div>
                        {processedImage && (
                            <div className="image-wrapper">
                                <h3>Ảnh sau khi xử lý</h3>
                                <img src={`http://127.0.0.1:8000/api/getProcessedImage`} alt="After processing" />
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageProcessingPage;
