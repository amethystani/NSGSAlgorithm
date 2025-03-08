#!/bin/bash

echo "================== Restarting YOLOv8 Object Detection App =================="

# Function to handle errors
handle_error() {
  echo "Error: $1"
  exit 1
}

# Stop any running processes
echo "Stopping any running processes..."
pkill -f "node.*server.js" || true
pkill -f "expo start" || true

# Create required directories
echo "Setting up required directories..."
mkdir -p Backend/Imginput
mkdir -p Backend/Imgoutput
mkdir -p Backend/models

# Check if ONNX headers exist, if not copy them
if [ ! -f Backend/include/onnxruntime_cxx_api.h ]; then
  echo "Copying ONNX Runtime headers..."
  cp -f Backend/onnxruntime-osx-arm64-1.15.0/include/onnxruntime_cxx_api.h Backend/include/ || handle_error "Failed to copy ONNX headers"
  cp -f Backend/onnxruntime-osx-arm64-1.15.0/include/onnxruntime_c_api.h Backend/include/ || handle_error "Failed to copy ONNX headers"
  cp -f Backend/onnxruntime-osx-arm64-1.15.0/include/onnxruntime_cxx_inline.h Backend/include/ || handle_error "Failed to copy ONNX headers"
  cp -f Backend/onnxruntime-osx-arm64-1.15.0/include/onnxruntime_session_options_config_keys.h Backend/include/ || handle_error "Failed to copy ONNX headers"
  cp -f Backend/onnxruntime-osx-arm64-1.15.0/include/onnxruntime_run_options_config_keys.h Backend/include/ || handle_error "Failed to copy ONNX headers"
fi

# Install backend dependencies
echo "Installing backend dependencies..."
cd Backend
npm install || handle_error "Failed to install backend dependencies"

# Start backend server in background
echo "Starting backend server..."
npm start &
BACKEND_PID=$!
cd ..

# Sleep to allow backend to initialize
echo "Waiting for backend to initialize..."
sleep 3

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd Frontend
npm install || handle_error "Failed to install frontend dependencies"

# Start frontend server
echo "Starting frontend server..."
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID; echo 'Shutting down servers...'" EXIT

# Wait for frontend to exit
wait 