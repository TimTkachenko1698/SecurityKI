#include "StableDetector.h"
#include "ArcFaceEmbedder.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <set>
#include <cmath>

bool StableDetector::initialize(const std::string& yoloConfig, const std::string& yoloWeights, 
                               const std::string& yoloFaceModel, const std::string& arcfaceModel) {
    std::cout << "=== Advanced AI Security System ===" << std::endl;
    std::cout << "Features: YOLOv4 + YOLOv8-Face + Tracking + Face Gallery" << std::endl;
    
    // Load YOLO model
    if (!personDetector.loadModel(yoloConfig, yoloWeights)) {
        std::cout << "Failed to load YOLO model!" << std::endl;
        return false;
    }
    std::cout << "YOLO model loaded successfully!" << std::endl;
    
    // Загружаем YOLOv8-face модель для детекции лиц
    if (!faceDetector.loadYOLOFaceModel(yoloFaceModel)) {
        std::cout << "Warning: YOLOv8-face model not loaded. Face detection disabled." << std::endl;
    } else {
        std::cout << "YOLOv8-face model loaded successfully!" << std::endl;
    }
    
    if (!arcfaceModel.empty() && !arcfaceEmbedder.loadModel(arcfaceModel)) {
        std::cout << "Warning: ArcFace model not loaded. Face recognition disabled." << std::endl;
    } else if (!arcfaceModel.empty()) {
        std::cout << "ArcFace model loaded successfully!" << std::endl;
    }
    
    return true;
}

void StableDetector::startDetection(int cameraIndex) {
    // Принудительно используем MSMF для USB камер
    cap.open(cameraIndex, cv::CAP_MSMF);
    if (!cap.isOpened()) {
        // Fallback на DSHOW
        cap.open(cameraIndex, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            std::cout << "Failed to connect to camera!" << std::endl;
            return;
        }
    }
    
    std::cout << "Connecting to USB camera..." << std::endl;
    runDetectionLoop();
}

void StableDetector::startDetection(const std::string& rtspUrl) {
    // Принудительно используем FFMPEG для RTSP
    cap.open(rtspUrl, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cout << "Failed to connect to RTSP camera!" << std::endl;
        return;
    }
    
    std::cout << "Connecting to: " << rtspUrl << std::endl;
    runDetectionLoop();
}

void StableDetector::stop() {
    isRunning = false;
    if (cap.isOpened()) {
        cap.release();
    }
    cv::destroyAllWindows();
}

void StableDetector::runDetectionLoop() {
    // Отключаем GStreamer backend чтобы избежать ошибок
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    
    // Optimize camera settings
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FPS, 25);
    
    // Create display window
    cv::namedWindow("Advanced AI Security", cv::WINDOW_NORMAL);
    cv::resizeWindow("Advanced AI Security", 1600, 900);
    
    std::cout << "System ready! Press ESC to exit." << std::endl;
    
    isRunning = true;
    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat previousFrame;
    
    while (isRunning) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        frameCount++;
        
        // Process YOLO every 10 frames (about 0.4 seconds at 25 FPS)
        if (frameCount % 10 == 0 && !processing.load()) {
            std::thread(&StableDetector::processYOLOAsync, this, frame.clone()).detach();
        }
        
        // Create extended display with gallery
        cv::Mat display = cv::Mat::zeros(frame.rows + 200, frame.cols + 300, CV_8UC3);
        
        // Apply face blurring for privacy
        cv::Mat blurredFrame = frame.clone();
        {
            std::lock_guard<std::mutex> lock(trackerMutex);
            auto trackedObjects = tracker.getObjects();
            std::vector<PersonData> currentPersons;
            
            for (const auto& pair : trackedObjects) {
                currentPersons.push_back(pair.second);
            }
            
            renderer.blurFaces(blurredFrame, currentPersons);
        }
        
        // Copy blurred video to display
        blurredFrame.copyTo(display(cv::Rect(0, 0, frame.cols, frame.rows)));
        
        // Draw tracked objects with silhouettes
        {
            std::lock_guard<std::mutex> lock(trackerMutex);
            auto trackedObjects = tracker.getObjects();
            std::vector<PersonData> currentPersons;
            
            for (const auto& pair : trackedObjects) {
                currentPersons.push_back(pair.second);
            }
            
            renderer.drawPersonBoxes(display, currentPersons, previousFrame);
        }
        
        // Draw visitor gallery
        {
            std::lock_guard<std::mutex> galleryLock(galleryMutex);
            auto visitors = gallery.getRecentVisitors();
            renderer.drawVisitorGallery(display, visitors);
        }
        
        // Performance info
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
        float fps = 0;
        if (elapsed > 0) {
            fps = frameCount / static_cast<float>(elapsed);
        }
        
        int activeCount = 0;
        int totalVisitors = 0;
        {
            std::lock_guard<std::mutex> lock(trackerMutex);
            activeCount = static_cast<int>(tracker.getObjects().size());
        }
        {
            std::lock_guard<std::mutex> galleryLock(galleryMutex);
            totalVisitors = gallery.getVisitorCount();
        }
        
        renderer.drawStats(display, activeCount, totalVisitors, fps, processing.load());
        
        cv::imshow("Advanced AI Security", display);
        
        // Store previous frame for movement detection
        previousFrame = frame.clone();
        
        // Exit on ESC
        if (cv::waitKey(1) == 27) {
            isRunning = false;
            break;
        }
    }
    
    stop();
    std::cout << "System shutdown complete." << std::endl;
}

void StableDetector::processYOLOAsync(cv::Mat frame) {
    if (processing.load()) return;
    processing = true;
    
    try {
        // Preprocess with shadow removal
        cv::Mat cleanFrame = personDetector.removeShadows(frame);
        
        // Detect persons with improved filtering
        std::vector<float> confidences;
        std::vector<cv::Rect> detections = personDetector.detectPersons(cleanFrame, confidences);
        
        // Упрощенная фильтрация - принимаем всех людей с уверенностью 50%+
        std::vector<cv::Rect> filteredDetections;
        std::vector<float> filteredConfidences;
        
        for (size_t i = 0; i < detections.size(); i++) {
            // Порог 40% для максимального обнаружения всех людей
            if (confidences[i] >= 0.4f) {
                filteredDetections.push_back(detections[i]);
                filteredConfidences.push_back(confidences[i]);
            }
        }
        
        detections = filteredDetections;
        confidences = filteredConfidences;
        
        // Update tracker
        {
            std::lock_guard<std::mutex> lock(trackerMutex);
            tracker.update(detections, confidences, frame, faceDetector, arcfaceEmbedder);
            tracker.updateFaceDescriptors(arcfaceEmbedder);
            
            // Update gallery
            {
                std::lock_guard<std::mutex> galleryLock(galleryMutex);
                auto trackedObjects = tracker.getObjects();
                
                for (const auto& pair : trackedObjects) {
                    const PersonData& person = pair.second;
                    // Фоткаем КАЖДЫЙ ID с лицом - пусть лучше будут дубликаты чем пропущенные люди
                    if (person.hasValidFace && !person.face.empty()) {
                        gallery.addVisitor(person);
                    }
                }
                
                // Remove old visitors
                gallery.removeOldVisitors(5);
            }
        }
        
    } catch (...) {}
    
    processing = false;
}

// CentroidTracker Implementation
std::map<int, PersonData> CentroidTracker::update(const std::vector<cv::Rect>& detections, 
                                                 const std::vector<float>& confidences,
                                                 const cv::Mat& frame, FaceDetector& faceDetector, ArcFaceEmbedder& arcfaceEmbedder) {
    
    // If no detections, increment disappeared counter
    if (detections.empty()) {
        std::vector<int> toRemove;
        for (auto& pair : disappeared) {
            pair.second++;
            if (pair.second > maxDisappeared) {
                toRemove.push_back(pair.first);
            }
        }
        for (int id : toRemove) {
            deregisterObject(id);
        }
        return objects;
    }

    // Calculate centroids for new detections
    std::vector<cv::Point2f> inputCentroids;
    for (const auto& rect : detections) {
        cv::Point2f centroid(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
        inputCentroids.push_back(centroid);
    }

    // If no existing objects, register all new detections
    if (objects.empty()) {
        for (size_t i = 0; i < inputCentroids.size(); i++) {
            registerObject(inputCentroids[i], detections[i], confidences[i], frame, faceDetector, arcfaceEmbedder);
        }
    } else {
        // Match existing objects with new detections
        std::vector<int> objectIDs;
        std::vector<cv::Point2f> objectCentroids;
        
        for (const auto& pair : objects) {
            objectIDs.push_back(pair.first);
            objectCentroids.push_back(pair.second.centroid);
        }

        // Calculate distance matrix
        std::vector<std::vector<float>> D(objectCentroids.size(), 
                                         std::vector<float>(inputCentroids.size()));
        
        for (size_t i = 0; i < objectCentroids.size(); i++) {
            for (size_t j = 0; j < inputCentroids.size(); j++) {
                float dx = objectCentroids[i].x - inputCentroids[j].x;
                float dy = objectCentroids[i].y - inputCentroids[j].y;
                D[i][j] = sqrt(dx * dx + dy * dy);
            }
        }

        // Simple assignment
        std::set<int> usedRows, usedCols;
        
        // Process matches
        for (size_t i = 0; i < objectIDs.size(); i++) {
            float minDist = maxDistance + 1;
            int bestMatch = -1;
            
            for (size_t j = 0; j < inputCentroids.size(); j++) {
                if (!usedCols.count(j) && D[i][j] < minDist) {
                    minDist = D[i][j];
                    bestMatch = j;
                }
            }
            
            if (bestMatch != -1 && minDist <= maxDistance) {
                // Update existing object
                int objectID = objectIDs[i];
                objects[objectID].centroid = inputCentroids[bestMatch];
                objects[objectID].bbox = detections[bestMatch];
                objects[objectID].confidence = confidences[bestMatch];
                objects[objectID].lastSeen = std::chrono::steady_clock::now();
                disappeared[objectID] = 0;
                
                // Обновляем лицо каждый кадр
                cv::Mat tempFace;
                cv::Rect tempFaceRect;
                bool currentlyHasFace = faceDetector.detectFace(frame, detections[bestMatch], tempFace, tempFaceRect);
                
                if (currentlyHasFace && !tempFace.empty()) {
                    // Проверяем качество нового лица
                    float newQuality = PersonData::calculateFaceQuality(tempFace, tempFaceRect);
                    if (newQuality > objects[objectID].faceQuality || !objects[objectID].hasValidFace) {
                        objects[objectID].face = tempFace.clone();
                        objects[objectID].faceRect = tempFaceRect;
                        objects[objectID].hasValidFace = true;
                        objects[objectID].faceQuality = newQuality;
                        
                        // Обновляем дескриптор
                        try {
                            objects[objectID].faceDescriptor = arcfaceEmbedder.generateEmbedding(tempFace);
                        } catch (...) {
                            objects[objectID].faceDescriptor = cv::Mat();
                        }
                    }
                } else {
                    // Лицо потеряно - очищаем
                    objects[objectID].hasValidFace = false;
                    objects[objectID].faceRect = cv::Rect();
                    objects[objectID].face = cv::Mat();
                    objects[objectID].faceQuality = 0.0f;
                }
                
                usedRows.insert(i);
                usedCols.insert(bestMatch);
            }
        }

        // Handle unmatched objects - быстро удаляем объекты
        std::vector<int> toRemove;
        for (size_t i = 0; i < objectIDs.size(); i++) {
            if (!usedRows.count(i)) {
                int objectID = objectIDs[i];
                disappeared[objectID]++;
                
                // Очищаем лицо немедленно
                if (objects.find(objectID) != objects.end()) {
                    objects[objectID].hasValidFace = false;
                    objects[objectID].faceRect = cv::Rect();
                    objects[objectID].face = cv::Mat();
                    objects[objectID].faceQuality = 0.0f;
                }
                
                // Удаляем объект через 10 кадров
                if (disappeared[objectID] > 10) {
                    toRemove.push_back(objectID);
                }
            }
        }
        
        // Безопасно удаляем объекты
        for (int id : toRemove) {
            deregisterObject(id);
        }

        // Register new objects - проверяем на похожие лица перед созданием нового ID
        for (size_t j = 0; j < inputCentroids.size(); j++) {
            if (!usedCols.count(j)) {
                // Проверяем нет ли уже такого человека по лицу
                cv::Mat newFace;
                cv::Rect newFaceRect;
                bool hasNewFace = faceDetector.detectFace(frame, detections[j], newFace, newFaceRect);
                
                int existingID = -1;
                if (hasNewFace && !newFace.empty()) {
                    // Генерируем дескриптор для нового лица
                    cv::Mat newDescriptor;
                    try {
                        newDescriptor = arcfaceEmbedder.generateEmbedding(newFace);
                    } catch (...) {}
                    
                    // ОЧЕНЬ осторожно ищем похожее лицо - лучше дубликат чем пропуск
                    if (!newDescriptor.empty()) {
                        for (const auto& pair : objects) {
                            if (pair.second.hasValidFace && !pair.second.faceDescriptor.empty()) {
                                float similarity = PersonData::cosineSimilarity(newDescriptor, pair.second.faceDescriptor);
                                // ОЧЕНЬ высокий порог 95% - только очень похожие лица
                                if (similarity > 0.95f) {
                                    // Дополнительная проверка по расстоянию
                                    float distance = cv::norm(inputCentroids[j] - pair.second.centroid);
                                    if (distance < 50.0f) { // Очень близко по позиции
                                        existingID = pair.first;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (existingID != -1) {
                    // Нашли похожего - переиспользуем ID
                    objects[existingID].centroid = inputCentroids[j];
                    objects[existingID].bbox = detections[j];
                    objects[existingID].confidence = confidences[j];
                    objects[existingID].lastSeen = std::chrono::steady_clock::now();
                    disappeared[existingID] = 0;
                    
                    if (hasNewFace && !newFace.empty()) {
                        float newQuality = PersonData::calculateFaceQuality(newFace, newFaceRect);
                        if (newQuality > objects[existingID].faceQuality) {
                            objects[existingID].face = newFace.clone();
                            objects[existingID].faceRect = newFaceRect;
                            objects[existingID].hasValidFace = true;
                            objects[existingID].faceQuality = newQuality;
                        }
                    }
                } else {
                    // Новый человек - создаем новый ID
                    registerObject(inputCentroids[j], detections[j], confidences[j], frame, faceDetector, arcfaceEmbedder);
                }
            }
        }
    }

    return objects;
}

void CentroidTracker::registerObject(const cv::Point2f& centroid, const cv::Rect& bbox, 
                                    float confidence, const cv::Mat& frame, FaceDetector& faceDetector, ArcFaceEmbedder& arcfaceEmbedder) {
    PersonData person;
    person.id = nextObjectID;
    person.centroid = centroid;
    person.bbox = bbox;
    person.confidence = confidence;
    person.lastSeen = std::chrono::steady_clock::now();
    person.disappearedFrames = 0;
    
    // Обнаруживаем лицо
    cv::Mat tempFace;
    cv::Rect tempFaceRect;
    person.hasValidFace = faceDetector.detectFace(frame, bbox, tempFace, tempFaceRect);
    
    if (person.hasValidFace && !tempFace.empty()) {
        person.face = tempFace.clone();
        person.faceRect = tempFaceRect;
        person.faceQuality = PersonData::calculateFaceQuality(person.face, person.faceRect);
        
        // Генерируем дескриптор для будущих сравнений
        try {
            person.faceDescriptor = arcfaceEmbedder.generateEmbedding(person.face);
        } catch (...) {
            person.faceDescriptor = cv::Mat();
        }
    } else {
        person.faceQuality = 0.0f;
    }
    
    objects[nextObjectID] = person;
    disappeared[nextObjectID] = 0;
    nextObjectID++;
}

void CentroidTracker::deregisterObject(int objectID) {
    if (objects.find(objectID) != objects.end()) {
        objects.erase(objectID);
    }
    if (disappeared.find(objectID) != disappeared.end()) {
        disappeared.erase(objectID);
    }
}