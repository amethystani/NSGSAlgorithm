# Neuro-Scheduling for Graph Segmentation (NSGS)

*An Event-Driven Approach to Parallel Image Processing*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++: 17](https://img.shields.io/badge/C++-17-orange.svg)](https://en.cppreference.com/w/cpp/17)
[![Platform: iOS/Android](https://img.shields.io/badge/Platform-iOS%20%7C%20Android-green)]()

## Overview

NSGS is a novel image segmentation framework inspired by neuromorphic computing principles. It models image regions as computational units (analogous to neurons) that communicate asynchronously via events ("spikes") triggered by local state changes exceeding adaptive thresholds. This event-driven paradigm facilitates inherent parallelism with minimal synchronization overhead, dynamically focusing computation on information-rich areas.

Key advantages:
- **Significant speedups (1.5-2.2x)** over traditional parallel approaches
- **Comparable segmentation accuracy** (measured by mIoU and Boundary F1 score)
- **Adaptive resource management** based on system load and thermal constraints
- **Efficient real-time processing** on resource-constrained platforms

## Project Structure

- **Frontend**: React Native (Expo) mobile application
- **Backend**: 
  - YOLOv8 baseline implementation
  - NSGS C++ implementation for efficient segmentation
  - Node.js API server

## Features

- **Event-Driven Computation**: Processing focuses on active image regions
- **Adaptive Thresholding**: Dynamically adjusts computational intensity based on system load
- **Minimal Synchronization**: Reduces overhead compared to synchronous methods
- **Thermal-Aware Processing**: Adjusts workload based on device temperature
- **Hybrid Deep Learning Integration**: Can work with CNN-based feature extraction
- **Mobile-Optimized**: Designed for resource-constrained devices

## Prerequisites

To build and run the project, you need:

- **Development Environment**:
  - C++17 compatible compiler (GCC 9+, Clang 10+, or MSVC 19.14+)
  - CMake 3.14+
  - Node.js 14+
  - npm or yarn
  - Expo CLI for frontend development

- **Dependencies**:
  - OpenCV 4.5.4+
  - ONNXRuntime 1.15+
  - Boost 1.74+ (for thread management)
  - React Native (Expo)

- **Optional**:
  - CUDA 11.0+ (for GPU acceleration)
  - Metal Performance Shaders (for iOS acceleration)
  - OpenCL 2.0+ (for cross-platform GPU support)

## Installation

### Clone the Repository

```bash
git clone https://github.com/username/nsgs.git
cd nsgs
```

### Backend Setup

1. Install dependencies (Ubuntu example):

```bash
# Core dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev

# OpenCV
sudo apt-get install -y libopencv-dev

# ONNXRuntime (optional)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-1.15.0.tgz
tar -xzf onnxruntime-linux-x64-1.15.0.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.15.0
```

2. Build the C++ application:

```bash
cd Backend
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

3. Install Node.js API server dependencies:

```bash
cd ../
npm install
```

4. Start the API server:

```bash
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
npm run dev
```

## Reproduction Steps

The following steps will guide you through reproducing the NSGS results presented in our paper:

### 1. Dataset Preparation

The benchmark datasets used in our experiments:

- [Cityscapes](https://www.cityscapes-dataset.com/) (validation set, 500 images)
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) (validation set, 1,449 images)
- [COCO-Stuff](https://github.com/nightrome/cocostuff) (subset of validation set, 5,000 images)
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) (validation set, 2,000 images)
- NeuroBoundary (our custom dataset, 2,500 images)

Download these datasets and place them in the `data` directory:

```bash
mkdir -p data/{cityscapes,pascal_voc,coco_stuff,ade20k,neuroboundary}
# Download datasets to their respective directories
```

### 2. Running the NSGS Algorithm

To run NSGS on a single image:

```bash
cd Backend/build
./nsgs_processor --input path/to/image.jpg --output path/to/output.png
```

Options:
- `--input`: Input image path
- `--output`: Output segmentation path
- `--model`: Model type (nsgs, yolov8, mobile_sam)
- `--threads`: Number of processing threads (default: auto)
- `--adaptive`: Enable adaptive threshold (0/1, default: 1)
- `--temp_aware`: Enable thermal awareness (0/1, default: 1)

For batch processing:

```bash
./nsgs_batch --dataset data/cityscapes --output results/cityscapes --model nsgs
```

### 3. Benchmarking

To reproduce the performance benchmarks:

```bash
cd Backend/build
./benchmark --dataset data/cityscapes --models nsgs,yolov8,mobile_sam --trials 5 --output benchmark_results.csv
```

This will generate a CSV file with execution times, memory usage, mIoU, and boundary F1 scores.

### 4. Visualization

To visualize results and performance metrics:

```bash
cd tools
python visualize_results.py --input ../benchmark_results.csv --output ../plots
```

This generates plots similar to those in our paper.

## Implementation Details

The NSGS implementation consists of several core components:

### Processing Elements (PEs)

Each PE (defined in `NeuronNode.h`) represents an image region with:
- Internal state (current segmentation label)
- Membrane potential (accumulating evidence)
- Firing threshold (adaptive)
- Refractory period
- Connections to adjacent PEs

```cpp
class NeuronNode {
private:
    int stateValue;           // Current segmentation label
    float potential;          // Accumulated evidence
    float threshold;          // Firing threshold
    bool refractoryPeriod;    // Prevents immediate re-activation
    std::vector<Connection> connections; // Links to adjacent nodes
    
    // Methods for handling potentials and firing
public:
    void updatePotential(float value, float weight);
    bool shouldFire() const;
    void fire(SpikeQueue& queue);
    // ...
};
```

### Event Queue

The event queue (implemented in `SpikeQueue.h`) uses a lock-free design:

```cpp
template <typename EventType>
class SpikeQueue {
private:
    // Lock-free ring buffer implementation
    std::vector<std::atomic<EventType*>> buffer;
    std::atomic<size_t> head;
    std::atomic<size_t> tail;
    
public:
    bool enqueue(EventType* event);
    EventType* dequeue();
    // ...
};
```

### Graph Construction

The segmentation graph is built using SLIC superpixels as the basis for PEs:

```cpp
// Simplified pseudocode
Graph constructGraph(const cv::Mat& image) {
    // Generate SLIC superpixels
    std::vector<Superpixel> superpixels = generateSLIC(image);
    
    // Create graph nodes
    Graph g;
    for (const auto& sp : superpixels) {
        g.addNode(createNeuronNode(sp));
    }
    
    // Add edges based on adjacency and feature similarity
    for (const auto& sp1 : superpixels) {
        for (const auto& sp2 : getNeighbors(sp1)) {
            float similarity = calculateSimilarity(sp1, sp2);
            g.addEdge(sp1.id, sp2.id, similarity);
        }
    }
    
    return g;
}
```

### Parallel Event Processing

The event processing occurs in worker threads, each handling events from the queue:

```cpp
// Worker thread pseudocode
void workerThread(SpikeQueue<Event>& queue, std::atomic<bool>& running) {
    while (running) {
        // Try to get an event
        Event* event = queue.dequeue();
        if (!event) {
            // No event available, try work stealing or backoff
            continue;
        }
        
        // Process the event
        NeuronNode* target = event->target;
        target->updatePotential(event->value, event->weight);
        
        // Check if target should fire
        if (target->shouldFire()) {
            target->fire(queue);
        }
        
        // Cleanup event
        delete event;
    }
}
```

## Performance

NSGS achieves significant performance improvements over traditional methods:

| Metric                  | NSGS  | YOLOv8 | Mobile SAM |
|-------------------------|-------|--------|------------|
| Execution Time (ms)     | 1342  | 9000+  | 7000+      |
| mIoU (%)                | 72.1  | 73.5   | 71.9       |
| Boundary F1             | 0.763 | 0.758  | 0.749      |
| Energy Consumption (J)  | 3.21  | 14.8   | 12.3       |
| Memory Usage (MB)       | 645   | 1240   | 980        |

For detailed performance analysis, see the [technical paper](Report/FinalReport.pdf).

## Mobile Application

The mobile app allows users to:
- Take photos or select from gallery
- Choose between YOLOv8 and NSGS processing
- Compare results side-by-side
- View processing time and resource usage
- Download processed images


## API Endpoints

- `POST /process`: Upload and process an image
  - Parameters: 
    - `image`: Image file
    - `model`: Processing model to use ('nsgs', 'yolov8', or 'mobile_sam')
    - `adaptive`: Enable adaptive thresholding (0/1)
  
- `GET /history`: Get processing history
  
- `GET /processed/:filename`: Retrieve processed image

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[MIT License](LICENSE)


## Contact

Animesh Mishra - am847@snu.edu.in

Project Link: [https://github.com/amethystani/nsgs](https://github.com/amethystani/nsgs) 
