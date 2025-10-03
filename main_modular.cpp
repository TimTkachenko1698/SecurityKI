#include "StableDetector.h"
#include <iostream>
#include <string>

int main() {
    StableDetector detector;
    
    // Initialize system
    if (!detector.initialize("yolov4.cfg", "yolov4.weights", 
                           "models/yolov8/yolov8_face.onnx", "models/arcface/arcfaceresnet100-8.onnx")) {
        std::cout << "Failed to initialize AI Security System!" << std::endl;
        return -1;
    }
    
    // Camera selection
    std::cout << "\nCamera Options:" << std::endl;
    std::cout << "1 - USB/Webcam" << std::endl;
    std::cout << "2 - IP Camera (RTSP)" << std::endl;
    std::cout << "Enter choice: ";
    
    int choice;
    std::cin >> choice;
    
    if (choice == 1) {
        detector.startDetection(0);
    } else {
        std::cout << "Enter RTSP URL (or press Enter for default): ";
        std::string rtspUrl;
        std::cin.ignore();
        std::getline(std::cin, rtspUrl);
        
        if (rtspUrl.empty()) {
            rtspUrl = "rtsp://admin:Snoel2025%21@192.168.2.102:554/unicast/c1/s0/live";
        }
        
        detector.startDetection(rtspUrl);
    }
    
    return 0;
}