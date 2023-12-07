import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import WelcomePage from './components/WelcomePage';
import InstructionsPage from './components/InstructionsPage';
import ImageProcessingPage from './components/ImageProcessingPage';

function App() {
  return (
    <Router>
      <div className="background">
        <nav className="container">
          <div className="row">
            <div className="button-container d-flex justify-content-between">
              <Link
                to="/instructions"
                className="button button--inline-block"
              >
                Hướng dẫn sử dụng
              </Link>
              <Link
                to="/image-processing"
                className="button button--inline-block"
              >
                Bắt đầu
              </Link>
            </div>
          </div>
        </nav>
        <Routes>
          <Route path="/" element={<WelcomePage />} />
          <Route path="/instructions" element={<InstructionsPage />} />
          <Route path="/image-processing" element={<ImageProcessingPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
