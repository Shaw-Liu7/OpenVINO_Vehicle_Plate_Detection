#include "video_detector.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 车牌字符映射
const char* const VideoDetector::items[] = {
    "0","1","2","3","4","5","6","7","8","9",
    "<Anhui>","<Beijing>","<Chongqing>","<Fujian>","<Gansu>","<Guangdong>",
    "<Guangxi>","<Guizhou>","<Hainan>","<Hebei>","<Heilongjiang>","<Henan>",
    "<HongKong>","<Hubei>","<Hunan>","<InnerMongolia>","<Jiangsu>","<Jiangxi>",
    "<Jilin>","<Liaoning>","<Macau>","<Ningxia>","<Qinghai>","<Shaanxi>",
    "<Shandong>","<Shanghai>","<Shanxi>","<Sichuan>","<Tianjin>","<Tibet>",
    "<Xinjiang>","<Yunnan>","<Zhejiang>","<police>",
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
};

// 构造函数：加载模型
VideoDetector::VideoDetector(const std::string& detection_model_path,
                             const std::string& recognition_model_path)
    : detection_model_path_(detection_model_path),
      recognition_model_path_(recognition_model_path) {

    ov::Core core;

    // 加载检测模型
    auto detection_model = core.read_model(detection_model_path_);
    ov::preprocess::PrePostProcessor ppp(detection_model);
    auto& inputInfo = ppp.input();
    inputInfo.tensor().set_element_type(ov::element::u8);
    inputInfo.tensor().set_layout({"NHWC"});
    detection_model_ = core.compile_model(ppp.build(), "CPU");

    // 加载识别模型
    auto recognition_model_raw = core.read_model(recognition_model_path_);
    ov::preprocess::PrePostProcessor p2p(recognition_model_raw);
    for (auto input : recognition_model_raw->inputs()) {
        if (input.get_shape().size() == 4)
            m_LprInputName = input.get_any_name();
        if (input.get_shape().size() == 2)
            m_LprSeqName = input.get_any_name();
    }
    auto& license_inputInfo = p2p.input(m_LprInputName);
    license_inputInfo.tensor().set_element_type(ov::element::u8);
    license_inputInfo.tensor().set_layout({"NCHW"});
    recognition_model_ = core.compile_model(p2p.build(), "CPU");

    recognition_request_ = recognition_model_.create_infer_request();
}

// 处理视频
void VideoDetector::processVideo(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        processFrame(frame);
        cv::imshow("Vehicle & Plate Detection", frame);
        if (cv::waitKey(1) == 27) break; // ESC退出
    }
    cap.release();
    cv::destroyAllWindows();
}

// 处理单帧
void VideoDetector::processFrame(cv::Mat& frame) {
    int ih = frame.rows;
    int iw = frame.cols;

    // 检测模型输入
    auto request = detection_model_.create_infer_request();
    ov::Shape input_shape = request.get_input_tensor().get_shape();
    size_t h = input_shape[1], w = input_shape[2], ch = input_shape[3];

    cv::Mat blob;
    cv::resize(frame, blob, cv::Size(w,h));

    auto allocator = std::make_shared<ov::AllocatorImpl>();
    ov::Tensor input_tensor(ov::element::u8, {1,h,w,ch}, blob.data);
    request.set_input_tensor(input_tensor);

    request.infer();

    // 获取输出
    ov::Tensor output = request.get_output_tensor();
    size_t num = output.get_shape()[2];
    size_t cnum = output.get_shape()[3];
    cv::Mat prob(num, cnum, CV_32F, (float*)output.data());

    int padding = 5;
    for (int i = 0; i < num; i++) {
        float conf = prob.at<float>(i,2);
        int label_id = prob.at<float>(i,1);
        if (conf > 0.75) {
            int x_min = static_cast<int>(prob.at<float>(i,3)*iw);
            int y_min = static_cast<int>(prob.at<float>(i,4)*ih);
            int x_max = static_cast<int>(prob.at<float>(i,5)*iw);
            int y_max = static_cast<int>(prob.at<float>(i,6)*ih);
            cv::Rect box(x_min,y_min,x_max-x_min,y_max-y_min);

            if (label_id == 2) { // 车牌
                cv::Rect plate_roi(box.x-padding, box.y-padding,
                                   box.width+2*padding, box.height+2*padding);
                cv::Mat temp_roi = frame(plate_roi);
                cv::Point txt_loc(box.x, box.y);
                fetch_plate_text(frame,temp_roi,txt_loc);
                cv::rectangle(frame, box, cv::Scalar(0,0,255),2);
            } else { // 车辆
                cv::rectangle(frame, box, cv::Scalar(0,255,255),2);
            }
        }
    }
}

// 车牌识别
void VideoDetector::fetch_plate_text(cv::Mat &frame, cv::Mat &temp_roi, cv::Point &txt_loc) {
    ov::Shape input_shape = recognition_request_.get_tensor(m_LprInputName).get_shape();
    size_t h = input_shape[2], w = input_shape[3], ch = input_shape[1];

    cv::Mat blob;
    cv::resize(temp_roi, blob, cv::Size(w,h));

    // 填充输入tensor
    auto input_tensor = recognition_request_.get_tensor(m_LprInputName);
    uchar* rp_data = input_tensor.data<uchar>();
    int image_size = w*h;
    for (size_t row=0; row<h; row++) {
        for (size_t col=0; col<w; col++) {
            for (size_t c=0; c<ch; c++) {
                rp_data[image_size*c + row*w + col] = blob.at<cv::Vec3b>(row,col)[c];
            }
        }
    }

    // 填充序列tensor
    ov::Tensor inputSeqtensor = recognition_request_.get_tensor(m_LprSeqName);
    std::fill(inputSeqtensor.data<float>(),
              inputSeqtensor.data<float>() + inputSeqtensor.get_shape()[0], 1.0f);

    recognition_request_.infer();

    ov::Tensor LprOutputTensor = recognition_request_.get_output_tensor();
    const auto out_data = LprOutputTensor.data<float>();
    std::string result;
    for (int i=0;i<LprOutputTensor.get_size();i++) {
        int val = int(out_data[i]);
        if (val==-1) break;
        result += items[val];
    }
    cv::putText(frame,result.c_str(),cv::Point(txt_loc.x-50,txt_loc.y-10),
                cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(255,255,0),2);
}
