#include "UIRenderer.h"
#include "PersonData.h"
#include <algorithm>

void UIRenderer::drawPersonBoxes(cv::Mat& frame, const std::vector<PersonData>& persons, 
                                const cv::Mat& previousFrame) {
    for (const auto& person : persons) {
        // Проверяем живой ли человек (есть ли микродвижения)
        bool isAlive = checkMovement(frame, previousFrame, person.bbox);
        
        // Рисуем контур силуэта для живых объектов
        if (isAlive) {
            drawSilhouetteContour(frame, person.bbox, cv::Scalar(255, 100, 0));
        } else {
            // Серый прямоугольник для статичных объектов
            cv::rectangle(frame, person.bbox, cv::Scalar(128, 128, 128), 1);
        }
        
        // Рисуем зеленый квадрат вокруг лица ТОЛЬКО если лицо видно СЕЙЧАС
        if (person.hasValidFace && person.faceRect.area() > 100 && !person.face.empty()) {
            // Проверяем что лицо в границах кадра
            cv::Rect safeFaceRect = person.faceRect & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safeFaceRect.area() > 100 && safeFaceRect.width > 20 && safeFaceRect.height > 20) {
                // Зеленый квадрат для видимого лица
                cv::rectangle(frame, safeFaceRect, cv::Scalar(0, 255, 0), 3);
                
                // Подпись "FACE DETECTED" с качеством
                int quality = static_cast<int>(person.faceQuality * 100);
                std::string faceLabel = "FACE Q:" + std::to_string(quality) + "%";
                cv::putText(frame, faceLabel, cv::Point(safeFaceRect.x, safeFaceRect.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
        
        // Подписываем ID, уверенность и статус лица
        std::string faceStatus = person.hasValidFace ? "FACE" : "BACK";
        std::string label = "ID:" + std::to_string(person.id) + 
                          " (" + std::to_string(static_cast<int>(person.confidence * 100)) + "%) " + faceStatus;
        
        cv::Scalar labelColor = person.hasValidFace ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255); // Зеленый для лица, оранжевый для спины
        cv::putText(frame, label, cv::Point(person.bbox.x, person.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, labelColor, 2);
        
        // Рисуем центроид (синяя точка)
        cv::circle(frame, person.centroid, 5, cv::Scalar(255, 0, 0), -1);
    }
}

void UIRenderer::drawVisitorGallery(cv::Mat& display, const std::vector<PersonData>& visitors) {
    // Создаем область галереи справа
    cv::Mat gallery = display(cv::Rect(display.cols - 300, 0, 300, display.rows));
    gallery.setTo(cv::Scalar(40, 40, 40));
    
    // Заголовок галереи
    cv::putText(gallery, "Unique Visitors", cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Общее количество посетителей
    std::string visitorCount = "Total: " + std::to_string(visitors.size());
    cv::putText(gallery, visitorCount, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    
    // Подпись о размытии лиц
    cv::putText(gallery, "(All detected people)", cv::Point(10, 80),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 150), 1);
    
    // Отображаем лица посетителей
    int y = 100;
    int maxFaces = (gallery.rows - 120) / 90;
    int startIdx = std::max(0, static_cast<int>(visitors.size()) - maxFaces);
    
    for (int i = startIdx; i < static_cast<int>(visitors.size()) && y + 80 < gallery.rows; i++) {
        cv::Rect faceRect(10, y, 80, 80);
        
        // Отображаем лицо (теперь все в галерее имеют лица)
        if (!visitors[i].face.empty()) {
            // Безопасное копирование с проверкой границ
            if (faceRect.x + faceRect.width <= gallery.cols && 
                faceRect.y + faceRect.height <= gallery.rows) {
                
                cv::Mat resizedFace;
                cv::resize(visitors[i].face, resizedFace, cv::Size(80, 80));
                
                if (resizedFace.type() == gallery.type()) {
                    resizedFace.copyTo(gallery(faceRect));
                } else {
                    cv::Mat convertedFace;
                    resizedFace.convertTo(convertedFace, gallery.type());
                    convertedFace.copyTo(gallery(faceRect));
                }
            }
        } else {
            // Серый квадрат для людей без лица
            cv::rectangle(gallery, faceRect, cv::Scalar(100, 100, 100), -1);
            cv::putText(gallery, "NO FACE", cv::Point(faceRect.x + 5, faceRect.y + 45),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1);
        }
        
        cv::rectangle(gallery, faceRect, cv::Scalar(255, 255, 255), 1);
        
        // Информация о времени (сколько секунд назад)
        auto now = std::chrono::steady_clock::now();
        auto timeAgo = std::chrono::duration_cast<std::chrono::seconds>(
            now - visitors[i].lastSeen).count();
        std::string timeText = std::to_string(timeAgo) + "s";
        cv::putText(gallery, timeText, cv::Point(100, y + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
        
        // ID посетителя, статус лица и качество
        std::string idText = "ID:" + std::to_string(visitors[i].id);
        std::string faceStatus = visitors[i].hasValidFace ? " FACE" : " BACK";
        if (visitors[i].hasValidFace) {
            int quality = static_cast<int>(visitors[i].faceQuality * 100);
            faceStatus += " Q:" + std::to_string(quality) + "%";
        }
        cv::putText(gallery, idText + faceStatus, cv::Point(100, y + 40),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
        
        y += 90;
    }
}

void UIRenderer::drawStats(cv::Mat& frame, int activeCount, int totalVisitors, float fps, bool processing) {
    // Количество активных людей в кадре
    std::string countText = "Active People: " + std::to_string(activeCount);
    cv::putText(frame, countText, cv::Point(10, 40),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    // Информация о производительности (FPS)
    std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fpsText, cv::Point(10, frame.rows - 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    // Статус обработки
    std::string status = processing ? "Processing..." : "Ready";
    cv::putText(frame, status, cv::Point(frame.cols - 150, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
}

void UIRenderer::blurFaces(cv::Mat& frame, const std::vector<PersonData>& persons) {
    // Размываем лица на основном видео для приватности
    for (const auto& person : persons) {
        if (person.hasValidFace && person.faceRect.area() > 0) {
            cv::Rect safeFaceRect = person.faceRect & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safeFaceRect.area() > 100) {
                // Применяем сильное размытие к области лица
                cv::Mat faceRegion = frame(safeFaceRect);
                cv::GaussianBlur(faceRegion, faceRegion, cv::Size(51, 51), 0);
            }
        }
    }
}

bool UIRenderer::checkMovement(const cv::Mat& currentFrame, const cv::Mat& previousFrame, 
                              const cv::Rect& bbox) {
    if (previousFrame.empty()) return true;
    
    cv::Rect safeROI = bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
    if (safeROI.area() < 2000) return true;
    
    try {
        cv::Mat currentRegion = currentFrame(safeROI);
        cv::Mat prevRegion = previousFrame(safeROI);
        
        cv::Mat currentGray, prevGray, diff;
        cv::cvtColor(currentRegion, currentGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(prevRegion, prevGray, cv::COLOR_BGR2GRAY);
        
        cv::absdiff(currentGray, prevGray, diff);
        cv::GaussianBlur(diff, diff, cv::Size(5, 5), 0);
        
        cv::Scalar meanDiff = cv::mean(diff);
        return (meanDiff[0] > 1.0 && meanDiff[0] < 20);
    } catch (...) {
        return true;
    }
}

void UIRenderer::drawSilhouetteContour(cv::Mat& frame, const cv::Rect& bbox, const cv::Scalar& color) {
    cv::Rect safeROI = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeROI.area() < 1000) {
        cv::rectangle(frame, bbox, color, 2);
        return;
    }
    
    try {
        cv::Mat personRegion = frame(safeROI);
        cv::Mat gray, mask;
        cv::cvtColor(personRegion, gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::threshold(blurred, mask, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty()) {
            double maxArea = 0;
            int maxIdx = -1;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxArea && area > 200) {
                    maxArea = area;
                    maxIdx = i;
                }
            }
            
            if (maxIdx >= 0) {
                std::vector<cv::Point> smoothContour;
                cv::approxPolyDP(contours[maxIdx], smoothContour, 2.0, true);
                
                for (auto& point : smoothContour) {
                    point.x += safeROI.x;
                    point.y += safeROI.y;
                }
                
                std::vector<std::vector<cv::Point>> contourVec = {smoothContour};
                cv::drawContours(frame, contourVec, -1, color, 2);
            } else {
                cv::rectangle(frame, bbox, color, 2);
            }
        } else {
            cv::rectangle(frame, bbox, color, 2);
        }
    } catch (...) {
        cv::rectangle(frame, bbox, color, 2);
    }
}