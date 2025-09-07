cmake_minimum_required(VERSION 3.15)

# 项目名
project(OpenVINO_Vehicle_Plate_Detection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 OpenVINO
find_package(InferenceEngine REQUIRED)
find_package(OpenVINO REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 源代码文件
set(SOURCES
    src/main.cpp
    src/video_detector.cpp
)

# 生成可执行文件
add_executable(vehicle_plate_detection ${SOURCES})

# 链接库
target_link_libraries(vehicle_plate_detection
    PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
)
