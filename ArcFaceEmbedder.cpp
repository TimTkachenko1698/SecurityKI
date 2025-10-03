#pragma warning(push)
#pragma warning(disable: 4244 4267 4018 4996 4305 4101)

#include "ArcFaceEmbedder.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

bool ArcFaceEmbedder::loadModel(const std::string& modelPath) {
    if (modelPath.empty()) {
        isLoaded = false;
        return false;
    }
    
    try {
        // Загружаем ArcFace ONNX модель через OpenCV DNN
        arcfaceNet = cv::dnn::readNetFromONNX(modelPath);
        
        // Пытаемся включить GPU ускорение
        try {
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                arcfaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                arcfaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                std::cout << "ArcFace: Using GPU acceleration" << std::endl;
            } else {
                arcfaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                arcfaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                std::cout << "ArcFace: Using CPU" << std::endl;
            }
        } catch (...) {
            // Fallback to CPU
            arcfaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            arcfaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "ArcFace: Fallback to CPU" << std::endl;
        }
        
        isLoaded = true;
        std::cout << "ArcFace model loaded successfully!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ArcFace model loading error: " << e.what() << std::endl;
        isLoaded = false;
        return false;
    } catch (...) {
        std::cerr << "Unknown ArcFace model loading error" << std::endl;
        isLoaded = false;
        return false;
    }
}

std::vector<float> ArcFaceEmbedder::getFaceEmbedding(const cv::Mat& face) {
    std::vector<float> embedding;
    if (!isLoaded || face.empty()) return embedding;
    
    try {
        // Предобрабатываем лицо для ArcFace
        cv::Mat preprocessed = preprocessFaceForArcFace(face);
        
        // Создаем blob для ArcFace (NCHW: 1x3x112x112)
        cv::Mat blob;
        cv::dnn::blobFromImage(preprocessed, blob, 1.0, cv::Size(112, 112), cv::Scalar(), true, false);
        arcfaceNet.setInput(blob);
        
        // Получаем эмбеддинг (512-мерный вектор)
        cv::Mat output = arcfaceNet.forward();
        
        // Конвертируем в std::vector<float>
        embedding.resize(output.total());
        std::memcpy(embedding.data(), output.data, output.total() * sizeof(float));
        
        // Нормализуем эмбеддинг (L2 нормализация)
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = sqrt(norm);
        
        if (norm > 0.0f) {
            for (float& val : embedding) {
                val /= norm;
            }
        }
        
    } catch (...) {
        embedding.clear();
    }
    
    return embedding;
}

cv::Mat ArcFaceEmbedder::generateEmbedding(const cv::Mat& face) {
    cv::Mat embedding;
    if (!isLoaded || face.empty()) return embedding;
    
    try {
        // Предобрабатываем лицо для ArcFace
        cv::Mat face_resized, face_normalized;
        cv::resize(face, face_resized, cv::Size(112, 112));
        face_resized.convertTo(face_normalized, CV_32F, 1.0/127.5, -1.0); // Normalize to [-1, 1]
        
        // Создаем blob для ArcFace (NCHW: 1x3x112x112)
        cv::Mat blob;
        cv::dnn::blobFromImage(face_normalized, blob, 1.0, cv::Size(112, 112), cv::Scalar(), true, false, CV_32F);
        
        arcfaceNet.setInput(blob);
        
        // Получаем эмбеддинг (512-мерный вектор)
        embedding = arcfaceNet.forward();
        
        // Проверяем размер эмбеддинга
        if (embedding.total() != 512) {
            std::cerr << "Invalid embedding size: " << embedding.total() << std::endl;
            return cv::Mat();
        }
        
        // Нормализуем эмбеддинг (L2 нормализация)
        cv::normalize(embedding, embedding, 1.0, 0.0, cv::NORM_L2);
        
    } catch (const std::exception& e) {
        std::cerr << "ArcFace embedding error: " << e.what() << std::endl;
        embedding = cv::Mat();
    } catch (...) {
        std::cerr << "Unknown ArcFace embedding error" << std::endl;
        embedding = cv::Mat();
    }
    
    return embedding;
}

cv::Mat ArcFaceEmbedder::preprocessFaceForArcFace(const cv::Mat& face) {
    cv::Mat face_resized, face_rgb, face_float;
    
    // Ресайз до 112x112 (стандартный размер для ArcFace)
    cv::resize(face, face_resized, cv::Size(112, 112));
    
    // Конвертируем BGR в RGB
    cv::cvtColor(face_resized, face_rgb, cv::COLOR_BGR2RGB);
    
    // Нормализация в диапазон [0, 1]
    face_rgb.convertTo(face_float, CV_32FC3, 1.0 / 255.0);
    
    return face_float;
}

#pragma warning(pop)