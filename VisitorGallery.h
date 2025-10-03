#pragma once
#include <vector>
#include <deque>
#include <opencv2/opencv.hpp>
#include "PersonData.h"

// Forward declarations
class FaceDetector;
class ArcFaceEmbedder;

class VisitorGallery {
private:
    std::deque<PersonData> visitors;
    
public:
    void addVisitor(const PersonData& person);
    void removeOldVisitors(int maxAgeMinutes = 5);
    std::vector<PersonData> getRecentVisitors() const;
    int getVisitorCount() const;
};