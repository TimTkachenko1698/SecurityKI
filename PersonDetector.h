#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class PersonDetector {
private:
    cv::dnn::Net net;
    bool isLoaded = false;
    
public:
    bool loadModel(const std::string& configPath, const std::string& weightsPath);
    std::vector<cv::Rect> detectPersons(const cv::Mat& frame, std::vector<float>& confidences);
    cv::Mat removeShadows(const cv::Mat& frame);
};