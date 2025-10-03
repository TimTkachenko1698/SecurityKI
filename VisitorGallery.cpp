#include "VisitorGallery.h"
#include "PersonData.h"
#include "FaceDetector.h"
#include "ArcFaceEmbedder.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <cmath>

void VisitorGallery::addVisitor(const PersonData& person) {
    // Фоткаем КАЖДЫЙ ID с лицом - просто и надежно
    if (!person.hasValidFace || person.face.empty()) {
        return;
    }
    
    // Проверяем только по ID - никаких умных проверок!
    for (auto& visitor : visitors) {
        if (visitor.id == person.id) {
            visitor.lastSeen = person.lastSeen;
            // Обновляем только если новое лучше
            float newQuality = PersonData::calculateFaceQuality(person.face, person.faceRect);
            if (newQuality > visitor.faceQuality) {
                visitor.face = person.face.clone();
                visitor.faceRect = person.faceRect;
                visitor.faceDescriptor = person.faceDescriptor;
                visitor.faceQuality = newQuality;
            }
            return;
        }
    }
    
    // Новый ID = новая запись в галерее
    PersonData newVisitor = person;
    newVisitor.face = person.face.clone();
    newVisitor.faceQuality = PersonData::calculateFaceQuality(newVisitor.face, newVisitor.faceRect);
    visitors.push_back(newVisitor);
}



void VisitorGallery::removeOldVisitors(int maxAgeMinutes) {
    auto now = std::chrono::steady_clock::now();
    
    // Удаляем старых посетителей (старше указанного времени)
    visitors.erase(
        std::remove_if(visitors.begin(), visitors.end(),
            [now, maxAgeMinutes](const PersonData& visitor) {
                return std::chrono::duration_cast<std::chrono::minutes>(
                    now - visitor.lastSeen).count() >= maxAgeMinutes;
            }),
        visitors.end()
    );
}

std::vector<PersonData> VisitorGallery::getRecentVisitors() const {
    // Возвращаем список последних посетителей
    return std::vector<PersonData>(visitors.begin(), visitors.end());
}

int VisitorGallery::getVisitorCount() const {
    // Возвращаем количество посетителей
    return static_cast<int>(visitors.size());
}

