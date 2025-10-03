#pragma warning(push)
#pragma warning(disable: 4244 4267 4018 4996 4305 4101)

#include "FaceDetector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <iostream>

bool FaceDetector::loadYOLOFaceModel(const std::string& modelPath) {
    try {
        // Пробуем загрузить как ONNX
        yoloFaceNet = cv::dnn::readNetFromONNX(modelPath);
        
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            yoloFaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            yoloFaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        
        isLoaded = true;
        return true;
    } catch (...) {
        // Если ONNX не работает, пробуем обычный readNet
        try {
            yoloFaceNet = cv::dnn::readNet(modelPath);
            isLoaded = true;
            return true;
        } catch (...) {
            isLoaded = false;
            return false;
        }
    }
}

std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& frame, float confThreshold) {
    std::vector<cv::Rect> faces;
    if (!isLoaded || frame.empty()) return faces;
    
    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        yoloFaceNet.setInput(blob);
        
        std::vector<cv::Mat> outputs;
        yoloFaceNet.forward(outputs);
        
        for (auto& output : outputs) {
            float* data = (float*)output.data;
            int rows = output.size[1];
            
            for (int i = 0; i < rows; ++i) {
                float confidence = data[4];
                if (confidence >= confThreshold) {
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    
                    int left = int((x - 0.5 * w) * frame.cols);
                    int top = int((y - 0.5 * h) * frame.rows);
                    int width = int(w * frame.cols);
                    int height = int(h * frame.rows);
                    
                    cv::Rect faceRect(left, top, width, height);
                    faceRect = faceRect & cv::Rect(0, 0, frame.cols, frame.rows);
                    
                    // Снижаем минимальную площадь лица для лучшей детекции на расстоянии
                    if (faceRect.area() > 100) { // Снижаем с 400 до 100
                        faces.push_back(faceRect);
                    }
                }
                data += output.size[2];
            }
        }
    } catch (...) {}
    
    return faces;
}

bool FaceDetector::detectFaceInROI(const cv::Mat& frame, const cv::Rect& personROI, 
                                  cv::Mat& faceOut, cv::Rect& faceRect) {
    if (!isLoaded) return false;
    
    try {
        // Создаем безопасную область для поиска лица в области человека
        cv::Rect safeROI = personROI & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeROI.area() < 500) return false; // Снижаем минимальную область
        
        // Вырезаем область человека
        cv::Mat personRegion = frame(safeROI);
        
        // Ищем лица в области человека с низким порогом
        std::vector<cv::Rect> faces = detectFaces(personRegion, 0.3f); // Снижаем порог до 30%
        
        if (!faces.empty()) {
            // Выбираем самое большое лицо
            cv::Rect bestFace = faces[0];
            for (const auto& face : faces) {
                if (face.area() > bestFace.area()) bestFace = face;
            }
            
            // Проверяем размер лица (снижаем минимум)
            if (bestFace.area() > 200) { // Снижаем с 400 до 200
                // Вырезаем и масштабируем лицо до 112x112 для ArcFace
                cv::Mat faceImg = personRegion(bestFace).clone();
                cv::resize(faceImg, faceOut, cv::Size(112, 112));
                
                // Глобальные координаты лица на кадре
                faceRect = cv::Rect(safeROI.x + bestFace.x, safeROI.y + bestFace.y, 
                                   bestFace.width, bestFace.height);
                return true;
            }
        }
    } catch (...) {}
    
    return false;
}

bool FaceDetector::detectFace(const cv::Mat& frame, const cv::Rect& personROI, 
                             cv::Mat& faceOut, cv::Rect& faceRect) {
    return detectFaceInROI(frame, personROI, faceOut, faceRect);
}

cv::Mat FaceDetector::preprocessFace(const cv::Mat& face) {
    // Преобразуем лицо в формат для ArcFace (RGB, 112x112, float)
    cv::Mat face_resized, face_rgb, face_float;
    cv::resize(face, face_resized, cv::Size(112, 112));
    cv::cvtColor(face_resized, face_rgb, cv::COLOR_BGR2RGB);
    face_rgb.convertTo(face_float, CV_32FC3, 1.0 / 255.0); // нормализация
    return face_float;
}

std::vector<float> FaceDetector::matToCHW(const cv::Mat& img) {
    // Преобразуем cv::Mat (HWC) в std::vector<float> CHW для ONNX
    std::vector<float> chw(3 * img.rows * img.cols);
    int channel_size = img.rows * img.cols;

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                chw[c * channel_size + i * img.cols + j] = img.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
    return chw;
}

#pragma warning(pop)