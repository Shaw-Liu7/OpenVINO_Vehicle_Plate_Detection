#Vehicle & License Plate Detection — OpenVINO项目

介绍
一个工业级、可部署、低延迟的车辆 + 车牌二阶段检测与识别 C++ 演示工程。基于 OpenVINO 推理引擎与 OpenCV，利用“零拷贝”内存分配、清晰的两阶段流水线（车辆检测 → 车牌识别）、和 OpenVINO 的预处理 API，将高准确率与低延迟结合起来——目标是在普通笔记本/台式 CPU 上也能拿到接近实时的推理体验，并且易于移植到 Intel GPU / VPU / 嵌入式平台。

项目亮点

两阶段流水线：用 vehicle-license-plate-detection-barrier-0106 做检测，检测到车牌再用 license-plate-recognition-barrier-0001 做识别，精度与稳健性兼顾。

零拷贝输入（Zero-copy）：自定义 SharedTensorAllocator 将 OpenCV 的 cv::Mat 直接绑定到 OpenVINO 的 ov::Tensor，避免额外内存拷贝，显著降低输入预处理开销。

OpenVINO 现代化 API：全程使用 ov::Core、ov::preprocess::PrePostProcessor、ov::CompiledModel 与 ov::InferRequest，代码清晰且易于扩展到不同设备（CPU/GPU/VPU）。

轻量级、易移植：C++17 + CMake，少依赖，方便集成到产线工程或边缘设备。

实用工程化细节：包含检测阈值、ROI padding、序列输入填充等工程细节，结果更稳健、误检更少。

面向部署：说明如何用 Model Zoo 下载模型、如何在不同设备上编译/运行、以及后续优化建议（量化、批量推理、线程池等）。


快速开始（Quick Start）
先决条件

OpenVINO（2022/2023 版本或更高）已安装并配置好环境。

OpenCV（建议 4.x）已安装。

CMake >= 3.16，C++17 编译器（g++ / clang / MSVC）。

模型准备

把以下模型的 IR 文件（.xml + .bin）放到 models/ 目录：

vehicle-license-plate-detection-barrier-0106（车辆+车牌检测）

license-plate-recognition-barrier-0001（车牌识别）

（建议使用 OpenVINO Model Zoo 下载并转换对应精度模型，或在 README 中提供下载脚本）

编译（示例）
git clone https://github.com/yourname/openvino-vehicle-plate-detection.git
cd openvino-vehicle-plate-detection

mkdir build && cd build
cmake ..
make -j$(nproc)

运行（示例）
# 默认 main.cpp 中使用了示例图片路径，后续可以改写为支持命令行参数
./vehicle_plate_detection ../data/car_test.png

核心方法与实现细节（技术说明）
1) 两阶段检测 + 识别（设计思路）

第 1 阶段：车辆与车牌检测模型输出一系列候选框（[label, confidence, x_min, y_min, x_max, y_max]）。当置信度高于阈值（示例中设为 0.75）时，把框绘制并作为后续处理依据。

第 2 阶段：若检测到 车牌（label_id==2），对车牌区域作扩展（padding）以包含更多上下文，然后送入车牌识别模型。识别模块输出为序列（按索引映射到 items[] 字符表），转换为车牌字符串显示在原图上。

2) 零拷贝输入（SharedTensorAllocator）

常规流程中，往往需要把 cv::Mat 的数据拷贝到 ov::Tensor，这一步在高帧率场景非常浪费。

本项目实现了 SharedTensorAllocator：直接把 cv::Mat::data 设为 ov::Tensor 的数据指针（只在安全场景下使用，注意生命周期管理）。这样可以避免内存复制，显著节省预处理时间与内存带宽，适用于单帧逐次推理的场景。

3) OpenVINO 预处理 API

使用 ov::preprocess::PrePostProcessor 指定输入元素类型与布局（NHWC / NCHW），保证模型在不同设备上的数据一致性与最优性能。

4) 识别序列处理

识别模型需要一个序列输入（m_LprSeqName），代码中默认填充为常数 1.0f，这在实际推理中起到占位与控制 decoder 步骤的作用（可在后续升级为 beam search、CTC 解码或自定义后处理）。

创新点 & 工程亮点

工业化的零拷贝输入方案：减少内存拷贝，输入预处理变得“瞬时”，对 CPU 边缘设备非常友好。

模块化两阶段设计：检测与识别解耦，既可保证识别精度又减少误识别（只有检测到车牌才做识别）。

OpenVINO 原生 API 应用：使用 PrePostProcessor 的正确示例，避免繁琐手工转置与类型转换，便于跨设备（CPU/GPU/VPU）迁移。

易迁移与扩展：你可以非常方便地替换检测模型（例如更快的单阶段模型）或把识别模型替换为更强的 CRNN/Transformer 模型，而不改动整体逻辑。

工程级细节：对 ROI 做 padding、对识别序列做默认填充、对输出做稳健截断——这些细节使结果在真实场景下更可靠。

与其他方案对比

与纯 Python + OpenCV + ONNX 的原型相比，本项目直接使用 OpenVINO 编译模型并完成推理，能更好利用 CPU 向量化指令与设备特性，延迟更低、部署更稳定。

与单阶段的端到端 OCR（把检测和识别合并）的方案比，两阶段方法在复杂场景下更稳健：检测把注意力先聚焦到车牌区域，识别模块在干净的 ROI 上工作，降低错识率。

与“不做零拷贝”的实现相比，本工程在输入预处理上做了优化，对于单帧到少量并发帧场景，能显著降低 CPU 占用与内存带宽压力。

总结：可工程化落地的推理模块，适合整合进后端服务、边缘设备或视频流处理管线。

可选的性能优化建议（Roadmap / Tips）

将模型转换为 FP16 / INT8（量化）以降低延迟与功耗（OpenVINO Post-Training Optimization/Quantization）。

使用批量推理或多线程推理池提升吞吐（当输入来自视频/相机）。

在 VPU（如 Intel Neural Compute Stick）或 GPU 上编译运行以获得更高帧率。

用更强的后处理（CTC 解码、beam search）替换当前简单阈值/映射策略，提升识别准确率。

常见问题（FAQ）

模型从哪儿来？
推荐从 OpenVINO Model Zoo 下载对应的模型 IR 文件（.xml + .bin），也可直接用 omz_downloader / omz_converter 获取。

如何支持视频或摄像头输入？
把 cv::imread 换为 cv::VideoCapture 循环读取帧，然后对每帧做相同的推理流程（推荐用线程池做并发推理）。

为什么使用 SharedTensorAllocator 有风险？
因为该方式要求 cv::Mat 在 ov::Tensor 生命周期内保持有效，需确保内存不会被提前释放或重分配。

改进建议

支持命令行参数（模型路径、设备、置信度阈值、输入来源）。

添加单元测试 / CI（GitHub Actions）自动构建。

提供 Dockerfile + demo 脚本，一键运行。

添加性能 benchmark 脚本输出 FPS / latency。

许可与声明

隐私声明：车牌属于个人隐私信息，使用时请遵守当地法律法规与隐私保护要求。

贡献 & 联系

很欢迎 PR、Issue、以及将你的优化（例如量化、RTSP 流集成、多线程优化）发回来！

