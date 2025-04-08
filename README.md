# YOLOv8 Object Detection App

A mobile application that allows users to detect objects in images using the YOLOv8 model. The app consists of a React Native (Expo) frontend and a C++ backend with a Node.js API server.

## Project Structure

- **Frontend**: React Native (Expo) mobile application
- **Backend**: YOLOv8 C++ application for object detection with Node.js API server

## Prerequisites

- Node.js (v14+)
- npm or yarn
- OpenCV 4.2+
- ONNXRuntime 1.15+
- C++ compiler
- CUDA (optional, for GPU acceleration)

## Getting Started

### Backend Setup

1. Build the C++ application:

```bash
cd Backend
sh build.sh
```

2. Install the Node.js API server dependencies:

```bash
cd Backend
npm install
```

3. Start the API server:

```bash
cd Backend
npm start
```

The server will run on port 3000 by default.

### Frontend Setup

1. Install dependencies:

```bash
cd Frontend
npm install
```

2. Update the API URL in `Frontend/api/imageProcessingApi.js` if needed (if your backend server is running on a different address).

3. Start the Expo development server:

```bash
cd Frontend
npm run dev
```

## Features

- Take photos or select images from the gallery
- Process images with YOLOv8 object detection or segmentation models
- View processing history
- Download processed images to the device gallery

## API Endpoints

- `POST /process`: Upload and process an image with the YOLOv8 model
- `GET /history`: Get a list of processed images
- `GET /processed/:filename`: Get a specific processed image

## How It Works

1. The user takes a photo or selects an image from the gallery in the mobile app
2. The image is uploaded to the backend server
3. The server saves the image to the input directory
4. The server runs the YOLOv8 C++ application to process the image
5. The processed image is saved to the output directory
6. The server returns the URL of the processed image to the frontend
7. The frontend displays the processed image to the user

## License

[MIT License](LICENSE)

# Neuro-Scheduling for Graph Segmentation (NSGS) Implementation Plan

This document outlines the high-level implementation plan for integrating NSGS, a neuromorphic-inspired, event-driven parallelism model, into our existing image processing system. The plan ensures that the current implementation remains untouched, offering a toggle on the mobile app to switch between the default model and our novel NSGS approach.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Implementation Strategy](#implementation-strategy)
3. [Backend Implementation Plan](#backend-implementation-plan)
4. [Frontend Implementation Plan](#frontend-implementation-plan)
5. [Testing and Validation Plan](#testing-and-validation-plan)
6. [Timeline and Milestones](#timeline-and-milestones)

## Project Overview

The NSGS approach introduces a spike-based, asynchronous update mechanism for graph-based image segmentation. Each node in the segmentation graph behaves like a neuron, firing updates only when its internal state crosses a dynamic threshold and propagating changes to neighbors in a non-blocking, event-driven fashion. This approach is specifically designed for graph-based image segmentation and is inspired by neuromorphic computing principles.

## Implementation Strategy

Our implementation strategy follows these key principles:

1. **Parallel Systems**: Implement NSGS as a separate module without modifying the existing image processing model
2. **Unified API Interface**: Create a common API layer that can route requests to either the traditional model or the NSGS module
3. **Toggle Mechanism**: Implement a user-friendly toggle in the mobile app to switch between models
4. **Performance Metrics**: Collect and display metrics to compare both approaches

## Backend Implementation Plan

### Phase 1: Core NSGS Module Development

1. **Create NSGS Directory Structure**:
   ```
   Backend/
   ├── src/
   │   ├── nsgs/
   │   │   ├── NsgsPredictor.h
   │   │   ├── NsgsPredictor.cpp
   │   │   ├── NeuronNode.h
   │   │   ├── NeuronNode.cpp
   │   │   ├── SpikeQueue.h
   │   │   ├── SpikeQueue.cpp
   │   │   └── GraphSegmentation.cpp
   ```

2. **Implement Core Data Structures**:
   - `NeuronNode`: Represents a node in the segmentation graph with state potential, threshold, and spike mechanics
   - `SpikeQueue`: Lock-free data structure for handling event propagation between nodes
   - `GraphSegmentation`: Graph construction and management for the segmentation process

3. **Implement NSGS Algorithm**:
   - Develop the event-driven, asynchronous update mechanism
   - Implement adaptive thresholding based on edge weights
   - Integrate thermal and power feedback mechanisms

### Phase 2: Integration with Existing Backend

1. **Create Unified Predictor Interface**:
   - Develop a common interface for both YOLOPredictor and NsgsPredictor
   - Implement a factory pattern to instantiate the appropriate predictor

2. **Extend Server API**:
   - Modify `server.js` to accept a model parameter that determines which processing approach to use
   - Create new endpoints for NSGS-specific operations if needed

3. **Optimize for Mobile Hardware**:
   - Integrate with mobile-specific scheduling APIs
   - Implement thermal sensing and adaptation

## Frontend Implementation Plan

### Phase 1: API Layer Updates

1. **Update API Client**:
   - Modify `imageProcessingApi.js` to support the model selection parameter
   - Add NSGS-specific API functions as needed

2. **Type Definitions**:
   - Update `types.ts` to include NSGS-specific types and parameters

### Phase 2: UI Implementation

1. **Add Model Selection Toggle**:
   - Update `index.tsx` to add a toggle switch between "Standard Model" and "NSGS Model"
   - Store user preference in local storage or context

2. **Performance Comparison UI**:
   - Create a visual comparison component to display processing time, accuracy, and energy usage
   - Implement a side-by-side view option to compare results

3. **Settings Integration**:
   - Add NSGS-specific settings in the settings screen
   - Provide educational information about the NSGS approach

## Testing and Validation Plan

1. **Unit Testing**:
   - Develop unit tests for all NSGS components
   - Test edge cases and boundary conditions

2. **Integration Testing**:
   - Verify seamless integration between the two processing approaches
   - Test the toggle functionality under various conditions

3. **Performance Benchmarking**:
   - Benchmark processing time, memory usage, and energy consumption
   - Compare segmentation quality between the two approaches
   - Test on various device types to ensure broad compatibility

4. **User Experience Testing**:
   - Conduct user tests to ensure the toggle is intuitive
   - Gather feedback on result quality and processing speed

## Timeline and Milestones

1. **Month 1: Research and Design**
   - Detailed algorithm design
   - System architecture planning
   - Implementation roadmap finalization

2. **Month 2: Core Implementation**
   - Develop core NSGS data structures and algorithms
   - Create initial integration points with existing system

3. **Month 3: Frontend and API Integration**
   - Implement API extensions
   - Develop UI toggle and comparison views

4. **Month 4: Testing and Optimization**
   - Comprehensive testing across devices
   - Performance optimization
   - Documentation and knowledge transfer

5. **Month 5: Beta Testing and Release**
   - Limited user testing
   - Final adjustments based on feedback
   - Full release with toggle feature

## Conclusion

The NSGS implementation provides a novel approach to image segmentation using neuromorphic-inspired, event-driven parallelism. By implementing this as a toggleable feature alongside our existing model, we enable users to choose the approach that best suits their needs while providing valuable comparative insights into this innovative technology. 