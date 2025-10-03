#pragma warning(push)
#pragma warning(disable: 4244 4267 4018 4996 4305 4101)

#include "PersonDetector.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>

bool PersonDetector::loadModel(const std::string& configPath, const std::string& weightsPath) {
    try {
        net = cv::dnn::readNetFromDarknet(configPath, weightsPath);
        
        // Пытаемся включить GPU ускорение
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        
        isLoaded = true;
        return true;
    } catch (...) {
        isLoaded = false;
        return false;
    }
}

std::vector<cv::Rect> PersonDetector::detectPersons(const cv::Mat& frame, std::vector<float>& confidences) {
    std::vector<cv::Rect> detections;
    confidences.clear();
    
    if (!isLoaded || frame.empty()) return detections;
    
    try {
        // Подготавливаем кадр для YOLO
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0,0,0), true, false);
        net.setInput(blob);
        
        // Прогоняем через нейросеть
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        
        // Обрабатываем результаты
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                cv::Mat scoresRow = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scoresRow, 0, &confidence, 0, &classIdPoint);
                
                // Детектируем только людей (класс 0) с высокой уверенностью
                if (confidence > 0.6 && classIdPoint.x == 0) {
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    cv::Rect bbox(left, top, width, height);
                    bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
                    
                    if (bbox.area() > 2000) { // Минимальный размер человека
                        boxes.push_back(bbox);
                        scores.push_back(static_cast<float>(confidence));
                    }
                }
            }
        }
        
        // Применяем подавление немаксимумов (убираем дубликаты)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.6, 0.4, indices);
        
        for (int idx : indices) {
            detections.push_back(boxes[idx]);
            confidences.push_back(scores[idx]);
        }
        
    } catch (...) {}
    
    return detections;
}

cv::Mat PersonDetector::removeShadows(const cv::Mat& frame) {
    cv::Mat result = frame.clone();
    
    try {
        cv::Mat gray, mask;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Вычисляем адаптивный порог для детекции теней
        cv::Scalar meanGray, stdGray;
        cv::meanStdDev(gray, meanGray, stdGray);
        double threshold = meanGray[0] - (stdGray[0] / 3.0);
        
        // Создаем маску теней
        cv::threshold(gray, mask, threshold, 255, cv::THRESH_BINARY_INV);
        
        // Морфологические операции для очистки маски
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        
        // Осветляем области теней
        cv::Mat brightened;
        frame.convertTo(brightened, -1, 1.3, 30);
        brightened.copyTo(result, mask);
        
    } catch (...) {
        return frame;
    }
    
    return result;
}

#pragma warning(pop)