cmake_minimum_required(VERSION 3.0.0)
project(yolov8_ort)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

include_directories("include/")
include_directories("src/")

add_executable(yolov8_ort
               src/utils.cpp
               src/yolov8Predictor.cpp
               src/nsgs/NeuronNode.cpp
               src/nsgs/SpikeQueue.cpp
               src/nsgs/GraphPartition.cpp
               src/nsgs/reconstructFromNeuralGraph.cpp
               src/nsgs/NsgsPredictor.cpp
               src/main.cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolov8_ort PRIVATE "${ONNXRUNTIME_DIR}/include")

target_compile_features(yolov8_ort PRIVATE cxx_std_17)
target_link_libraries(yolov8_ort ${OpenCV_LIBS})


if (WIN32)
    target_link_libraries(yolov8_ort "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (APPLE)
    target_link_libraries(yolov8_ort "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif (UNIX)
    target_link_libraries(yolov8_ort "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

