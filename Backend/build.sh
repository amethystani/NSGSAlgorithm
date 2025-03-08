#!/bin/bash
# Clean up
rm -rf build

# Create output directory if it doesn't exist
mkdir -p Imgoutput

# Get absolute path to onnxruntime directory
BACKEND_DIR=$(pwd)
ONNX_RUNTIME_DIR="${BACKEND_DIR}/onnxruntime-osx-arm64-1.15.0"

# Create build directory and configure with CMake
mkdir -p build
cd build
cmake .. -DONNXRUNTIME_DIR="${ONNX_RUNTIME_DIR}" -DCMAKE_BUILD_TYPE=Debug

# Build the project
make

# Go back to Backend directory
cd ..

# Output success message
if [ -f "./build/yolov8_ort" ]; then
  echo "Build successful! Executable created at ./build/yolov8_ort"
else
  echo "Build failed. Executable not created."
  exit 1
fi

# Don't run the model automatically, as it might fail without proper input
# Let the user run it manually
# sh run.sh