import React from 'react';
import './style.css';
import batdauImg from './image/batdau.png';
import taianhImg from './image/taianh.png';
import startImg from './image/start.png';
import appImg from './image/app.png';
import interfaceImg from './image/interface.png';

const InstructionsPage = () => {
    return (
        <div className='container'>
            <h2 className='text'>Hướng dẫn sử dụng</h2>
            <div className='instruct'><strong>Bước 1:</strong> Mở Project sau đó mở terminal chạy câu lệnh "uvicorn App:app --reload", sau đó mở 1 terminal mới chạy 2 câu lệnh sau "cd .\patient-recard-app\", và "npm start". Chờ 1 lúc chương trình sẽ mở lên.</div>
            <img src={appImg} alt="Ảnh minh họa bắt đầu" />
            <img src={interfaceImg} alt="Ảnh minh họa bắt đầu" />
            <div className='instruct'><strong>Bước 2: </strong>Chọn qua mục bắt đầu để có thể bắt đầu sử dụng giao diện.</div>
            <img src={batdauImg} alt="Ảnh minh họa bắt đầu" />
            <div className='instruct'><strong>Bước 3:</strong> Ấn vào nút "Chọn tệp" và chọn ảnh của bạn sau đó ấn nút "Open" để có thể tải ảnh lên.</div>
            <img src={taianhImg} alt="Ảnh minh họa tải ảnh" />
            <div className='instruct'><strong>Bước 4:</strong> Ấn nút "Bắt đầu" để có thể xử lý ảnh.</div>
            <img src={startImg} alt="Ảnh minh họa Start" />
        </div>
    );
};

export default InstructionsPage;
