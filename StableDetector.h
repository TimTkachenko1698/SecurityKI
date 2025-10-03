#pragma once
#include "PersonDetector.h"
#include "FaceDetector.h"
#include "ArcFaceEmbedder.h"
#include "VisitorGallery.h"
#include "UIRenderer.h"
#include "PersonData.h"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>

// Centroid Tracker Class
class CentroidTracker {
private:
    int nextObjectID;
    std::map<int, PersonData> objects;
    std::map<int, int> disappeared;
    int maxDisappeared;
    float maxDistance;

public:
    CentroidTracker(int maxDisappeared = 50, float maxDistance = 100.0f) 
        : nextObjectID(0), maxDisappeared(maxDisappeared), maxDistance(maxDistance) {}
    
    std::map<int, PersonData> getObjects() const { return objects; }
    
    void updateFaceDescriptors(ArcFaceEmbedder& arcfaceEmbedder) {
        for (auto& pair : objects) {
            PersonData& person = pair.second;
            if (person.hasValidFace && person.faceDescriptor.empty()) {
                try {
                    person.faceDescriptor = arcfaceEmbedder.generateEmbedding(person.face);
                } catch (...) {
                    // ArcFace failed, skip
                }
            }
        }
    }
    
    std::map<int, PersonData> update(const std::vector<cv::Rect>& detections, 
                                    const std::vector<float>& confidences,
                                    const cv::Mat& frame, FaceDetector& faceDetector, ArcFaceEmbedder& arcfaceEmbedder);
    
private:
    void registerObject(const cv::Point2f& centroid, const cv::Rect& bbox, 
                       float confidence, const cv::Mat& frame, FaceDetector& faceDetector, ArcFaceEmbedder& arcfaceEmbedder);
    void deregisterObject(int objectID);
};

class StableDetector {
private:
    PersonDetector personDetector;
    FaceDetector faceDetector;
    ArcFaceEmbedder arcfaceEmbedder;
    VisitorGallery gallery;
    UIRenderer renderer;
    CentroidTracker tracker;
    
    std::atomic<bool> isRunning{false};
    std::atomic<bool> processing{false};
    cv::VideoCapture cap;
    
    std::mutex trackerMutex;
    std::mutex galleryMutex;
    
public:
    bool initialize(const std::string& yoloConfig, const std::string& yoloWeights, 
                   const std::string& yoloFaceModel, const std::string& arcfaceModel);
    void startDetection(int cameraIndex = 0);
    void startDetection(const std::string& rtspUrl);
    void stop();
    
private:
    void runDetectionLoop();
    void processYOLOAsync(cv::Mat frame);
};