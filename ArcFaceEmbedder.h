#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

// Простая реализация без ONNX Runtime для начала
class ArcFaceEmbedder {
private:
    bool isLoaded = false;
    cv::dnn::Net arcfaceNet;
    
public:
    bool loadModel(const std::string& modelPath);
    std::vector<float> getFaceEmbedding(const cv::Mat& face);
    cv::Mat generateEmbedding(const cv::Mat& face);
    
private:
    cv::Mat preprocessFaceForArcFace(const cv::Mat& face);
};