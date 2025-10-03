#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class FaceDetector {
private:
    cv::dnn::Net yoloFaceNet;
    cv::CascadeClassifier faceCascade;
    bool isLoaded = false;
    bool useYOLO = false;
    
public:
    bool loadYOLOFaceModel(const std::string& modelPath);
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame, float confThreshold = 0.5f);
    bool detectFaceInROI(const cv::Mat& frame, const cv::Rect& personROI, 
                        cv::Mat& faceOut, cv::Rect& faceRect);
    bool detectFace(const cv::Mat& frame, const cv::Rect& personROI, 
                   cv::Mat& faceOut, cv::Rect& faceRect);
    
    // Вспомогательные функции для предобработки
    cv::Mat preprocessFace(const cv::Mat& face);
    std::vector<float> matToCHW(const cv::Mat& img);
};