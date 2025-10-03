#pragma once
#include "PersonData.h"
#include <opencv2/opencv.hpp>
#include <vector>

class UIRenderer {
public:
    void drawPersonBoxes(cv::Mat& frame, const std::vector<PersonData>& persons, 
                        const cv::Mat& previousFrame);
    void drawVisitorGallery(cv::Mat& display, const std::vector<PersonData>& visitors);
    void drawStats(cv::Mat& frame, int activeCount, int totalVisitors, float fps, bool processing);
    void blurFaces(cv::Mat& frame, const std::vector<PersonData>& persons);
    
private:
    bool checkMovement(const cv::Mat& currentFrame, const cv::Mat& previousFrame, 
                      const cv::Rect& bbox);
    void drawSilhouetteContour(cv::Mat& frame, const cv::Rect& bbox, const cv::Scalar& color);
};