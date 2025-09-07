#include "video_detector.hpp"
#include <iostream>

int main() {
    std::string detection_model = "models/vehicle-license-plate-detection-barrier-0106.xml";
    std::string recognition_model = "models/license-plate-recognition-barrier-0001.xml";

    VideoDetector detector(detection_model, recognition_model);

    std::string video_path = "videos/test_video.mp4"; // 或摄像头编号 "0"
    detector.processVideo(video_path);

    return 0;
}