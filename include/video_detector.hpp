#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class VideoDetector {
public:
    // 构造函数：传入两个模型路径
    VideoDetector(const std::string& detection_model_path,
                  const std::string& recognition_model_path);

    // 处理视频文件
    void processVideo(const std::string& video_path);

    // 处理单帧图像
    void processFrame(cv::Mat& frame);

private:
    void fetch_plate_text(cv::Mat &frame, cv::Mat &temp_roi, cv::Point &txt_loc);

    std::string detection_model_path_;
    std::string recognition_model_path_;

    ov::CompiledModel detection_model_;
    ov::CompiledModel recognition_model_;
    ov::InferRequest recognition_request_;

    std::string m_LprInputName;
    std::string m_LprSeqName;

    static const char* const items[];
};
