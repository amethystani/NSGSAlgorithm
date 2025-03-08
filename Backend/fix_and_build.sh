#!/bin/bash

echo "Copying ONNX Runtime headers to include directory..."
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_cxx_api.h include/
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_c_api.h include/
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_cxx_inline.h include/
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_session_options_config_keys.h include/
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_run_options_config_keys.h include/

echo "Creating necessary directories..."
mkdir -p build
mkdir -p Imgoutput
mkdir -p models

echo "Configuring and building the project..."
cd build
cmake .. -DONNXRUNTIME_DIR=$(pwd)/../onnxruntime-osx-arm64-1.15.0 -DCMAKE_BUILD_TYPE=Debug
make

# Go back to Backend directory
cd ..

# Check if build was successful
if [ -f "./build/yolov8_ort" ]; then
  echo "Build successful! Executable created at ./build/yolov8_ort"
  echo "You can now run the model with: ./build/yolov8_ort -m ./models/yolov8m.onnx -i ./Imginput -o ./Imgoutput -c ./models/coco.names -x m"
else
  echo "Build failed. Executable not created."
  exit 1
fi 