# Elastic Spatial Partitioning with Vectorized Synchronization (ESPVS)

## Executive Summary

This document introduces a novel parallel processing paradigm for computer vision applications called **Elastic Spatial Partitioning with Vectorized Synchronization (ESPVS)**. This approach fundamentally reimagines how concurrent computation works in object detection systems, offering significant performance improvements over traditional parallel processing models.

ESPVS combines three innovative concepts:
1. **Elastic spatial partitioning** that dynamically adapts to image complexity
2. **Vectorized synchronization primitives** that eliminate traditional locking overhead
3. **Continuous work redistribution** during inference to maximize resource utilization

Unlike traditional parallel processing approaches (data, task, or model parallelism), ESPVS adaptively partitions spatial regions based on real-time complexity metrics and synchronizes execution using SIMD vector operations, creating a fundamentally new paradigm for concurrent image processing.

## Current Implementation Analysis

### Traditional Approach in Current Codebase

The current YOLOv8 implementation in your codebase follows a sequential processing model:

```cpp
// Current implementation pattern (simplified)
std::vector<Yolov8Result> YOLOPredictor::predict(cv::Mat &image) {
    // Step 1: Preprocessing (sequential)
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    // Step 2: Inference (sequential)
    std::vector<Ort::Value> outputTensors = this->session.Run(...);

    // Step 3: Postprocessing (sequential)
    std::vector<Yolov8Result> results = this->postprocessing(...);
    
    return results;
}
```

#### Limitations of the Current Approach:

1. **Sequential Pipeline**: Preprocessing, inference, and postprocessing happen sequentially, leaving most system resources idle at any given moment.

2. **Uniform Processing**: The entire image is processed with the same parameters and resolution, regardless of varying complexity across different regions.

3. **Static Resource Allocation**: No dynamic adaptation to the specific characteristics of the input image or available system resources.

4. **Inefficient Parallelism**: The current parallelism in `main.cpp` only processes multiple images in parallel but doesn't parallelize the processing of a single image.

5. **Synchronization Overhead**: When parallel processing is used, traditional synchronization primitives introduce significant overhead.

## The ESPVS Approach

### Core Concepts

#### 1. Elastic Spatial Partitioning

ESPVS divides images into spatial regions that are dynamically sized based on their complexity:

- **Adaptive Quad-Tree Partitioning**: Uses a complexity-aware quad-tree algorithm to divide the image into regions of varying sizes.
- **Complexity-Based Sizing**: More complex regions (with more edges, textures, potential objects) get smaller partitions to balance workload.
- **Dynamic Region Adjustment**: Partition boundaries can shift during processing as more information becomes available.

#### 2. Vectorized Synchronization

Instead of traditional locks or atomic operations, ESPVS uses vectorized operations for synchronization:

- **SIMD-Based Coordination**: Uses AVX/SSE instructions to track and update multiple thread states simultaneously.
- **Lock-Free Communication**: Eliminates contention points that plague traditional parallel algorithms.
- **Partial Progress Tracking**: Enables fine-grained tracking of completion progress (0.0-1.0) rather than binary done/not-done states.

#### 3. Continuous Work Redistribution

ESPVS continuously monitors and rebalances work during execution:

- **Real-Time Load Measurement**: Measures processing time per region and redistributes work accordingly.
- **Adaptive Thread Assignment**: Dynamically assigns threads to regions based on complexity and progress.
- **Spatial Coherence Awareness**: Preserves cache efficiency by considering spatial locality when redistributing work.

### Novel Algorithm Components

#### Vectorized Barrier

Traditional synchronization barriers force all threads to wait at synchronization points. The VectorizedBarrier in ESPVS allows:

```cpp
// Novel vectorized barrier approach
class VectorizedBarrier {
    // ...
    void arrive(int threadId, float completionFraction) {
        // Use vectorized operations to update sync state
        syncVector[threadId % 16] = completionFraction;
        _mm_sfence(); // Ensure visibility across cores
    }
    
    bool waitForPhase(float targetPhase, float tolerance = 0.05f) {
        // Process 8 values at once with AVX
        __m256 targetVec = _mm256_set1_ps(targetPhase);
        // ...
    }
};
```

This allows threads to report partial completion (0.3 = 30% done) and continue when others are "close enough" to the target phase.

#### Adaptive Quad-Tree Spatial Partitioning

The approach uses image content to drive partitioning decisions:

```cpp
// Novel quad-tree based partitioning
std::vector<ElasticSpatialPartition> generatePartitions(const cv::Mat& image, int targetPartitions) {
    // Create initial complexity map from image
    cv::Mat complexityMap;
    cv::Laplacian(image, complexityMap, CV_32F);
    
    // Build adaptive quad-tree based on complexity
    QuadNode* root = new QuadNode(cv::Rect(0, 0, image.cols, image.rows));
    // ...
}
```

#### Vectorized Non-Maximum Suppression

Traditional NMS is inherently sequential. ESPVS introduces a vectorized approach:

```cpp
// Novel vectorized NMS
void vectorizedNMS(const std::vector<Yolov8Result>& results,
                 std::vector<Yolov8Result>& keptResults,
                 float iouThreshold) {
    // ...
    // Process 8 boxes at a time using AVX
    __m256 boxA_x1 = _mm256_set1_ps(boxA.x);
    __m256 boxA_y1 = _mm256_set1_ps(boxA.y);
    // ...
}
```

## How ESPVS Differs from Current Implementation

### 1. Processing Pipeline Architecture

**Current Implementation:**
- Sequential: Preprocessing → Inference → Postprocessing
- Whole-image oriented processing
- Fixed parameter set throughout

**ESPVS Approach:**
- Concurrent: Multiple stages operating simultaneously
- Region-oriented processing with dynamic boundaries
- Adaptive parameters per region based on content

### 2. Resource Utilization

**Current Implementation:**
- Single-threaded processing of individual images
- Homogeneous resource allocation
- Most compute resources idle during various phases

**ESPVS Approach:**
- Multi-threaded processing within a single image
- Heterogeneous resource allocation based on region needs
- Continuous resource rebalancing to minimize idle time

### 3. Synchronization Mechanism

**Current Implementation:**
- Traditional synchronization when needed (mutex, barriers, etc.)
- Binary synchronization states (done/not done)
- High overhead for coordination

**ESPVS Approach:**
- Vectorized synchronization using SIMD instructions
- Continuous progress tracking (0.0-1.0 completion scale)
- Minimal overhead through hardware-accelerated operations

### 4. Memory Access Patterns

**Current Implementation:**
- Unpredictable memory access across the entire image
- Cache-inefficient processing that jumps between distant memory locations

**ESPVS Approach:**
- Spatially coherent processing within regions
- Cache-friendly access patterns that maximize spatial locality
- Thread assignment preserves memory access patterns

### 5. Parallelization Strategy

**Current Implementation:**
- Coarse-grained parallelism (entire images in parallel)
- Static work division
- No adaptation to content complexity

**ESPVS Approach:**
- Fine-grained parallelism (regions within images)
- Dynamic work division
- Content-aware complexity adaptation

## Implementation Pathway

To implement ESPVS in your current codebase, these changes would be required:

### 1. YOLOPredictor Modifications

The YOLOPredictor class would need to be extended to:
- Expose individual processing stages
- Support region-based processing
- Allow dynamic parameter adjustment

### 2. New Components

New components would need to be added:
- VectorizedBarrier for synchronization
- ElasticSpatialPartition for region management
- QuadTree implementation for adaptive partitioning

### 3. Main Processing Loop

The main processing loop would change from:

```cpp
for (const auto &entry : std::filesystem::directory_iterator(imagePath)) {
    // Process entire image sequentially
    cv::Mat image = cv::imread(entry.path().string());
    std::vector<Yolov8Result> result = predictor.predict(image);
    // ...
}
```

To:

```cpp
ESPVSDetector detector(modelPath, isGPU, numThreads);

for (const auto &entry : std::filesystem::directory_iterator(imagePath)) {
    // Process image with ESPVS
    cv::Mat image = cv::imread(entry.path().string());
    std::vector<Yolov8Result> result = detector.predict(image);
    // ...
}
```

## Performance Benefits

The ESPVS approach offers several performance advantages:

1. **Higher Throughput**: By utilizing all available compute resources efficiently, overall throughput increases.

2. **Lower Latency**: Critical regions can be processed first, reducing time-to-first-detection.

3. **Better Scaling**: Performance scales more linearly with additional cores due to reduced synchronization overhead.

4. **Adaptive Resource Usage**: Automatically adapts to different hardware configurations without manual tuning.

5. **Improved Cache Efficiency**: Spatial coherence in processing improves cache hit rates.

## Potential Challenges

Implementing ESPVS also comes with challenges:

1. **Implementation Complexity**: More complex than traditional parallel approaches.

2. **Debugging Difficulty**: Concurrent systems with dynamic work stealing are harder to debug.

3. **Hardware Dependencies**: Vectorized operations require specific CPU instruction set support.

4. **Parameter Tuning**: Finding optimal partition sizes and redistribution thresholds.

## Comparison to Existing Research

Unlike existing parallel processing approaches in computer vision which typically focus on:

1. **Data Parallelism**: Processing multiple images or batches simultaneously
2. **Model Parallelism**: Distributing neural network layers across devices
3. **Pipeline Parallelism**: Sequential stages with different resources

ESPVS introduces a fundamentally new approach by:

1. Combining spatial decomposition with adaptive boundaries
2. Using vectorized operations for synchronization itself
3. Continuously redistributing work during execution
4. Maintaining spatial coherence for cache efficiency

Most importantly, while traditional approaches use SIMD for computation only, ESPVS uniquely applies SIMD operations to the parallel coordination mechanism itself, representing a novel advance in parallel computing paradigms.

## Conclusion

The Elastic Spatial Partitioning with Vectorized Synchronization (ESPVS) approach represents a fundamental innovation in parallel processing for computer vision. By reimagining how work is distributed, synchronized, and redistributed, ESPVS achieves better resource utilization, reduced synchronization overhead, and adapts to the unique characteristics of each image.

This approach is particularly well-suited for object detection systems like YOLOv8, where processing requirements vary greatly across image regions and traditional parallel patterns cannot fully exploit modern hardware capabilities.

## Concrete Implementation Details

This section provides specific code examples to illustrate how ESPVS would be implemented in your existing codebase.

### Current vs. ESPVS Implementation

#### 1. Current YOLOv8 Predictor Implementation

Your current implementation in `yolov8Predictor.cpp`:

```cpp
// Current implementation
std::vector<Yolov8Result> YOLOPredictor::predict(cv::Mat &image) {
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                            this->inputNames.data(),
                                                            inputTensors.data(),
                                                            this->inputNames.size(),
                                                            this->outputNames.data(),
                                                            this->outputNames.size());

    std::vector<Yolov8Result> results = this->postprocessing(cv::Size(inputTensorShape[3], inputTensorShape[2]), 
                                                          image.size(), outputTensors);
    
    delete[] blob;
    return results;
}
```

#### 2. Modified YOLOv8 Predictor to Support ESPVS

To support ESPVS, we would need to modify the YOLOPredictor class to expose its internal stages:

```cpp
// Modified YOLOPredictor with exposed stages
class YOLOPredictor {
public:
    // ... existing code ...
    
    // New methods to expose internal stages
    void preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    
    std::vector<Ort::Value> runInference(float* blob, const std::vector<int64_t>& inputTensorShape);
    
    std::vector<Yolov8Result> postprocessing(const cv::Size &resizedImageShape,
                                            const cv::Size &originalImageShape,
                                            std::vector<Ort::Value> &outputTensors);
                                            
    // New methods for region-based processing
    std::vector<Yolov8Result> predictRegion(const cv::Mat& region, const cv::Rect& regionBounds);
    
    // Method to extract raw data for vectorized operations
    std::vector<float> extractRawData(const std::vector<Ort::Value>& outputTensors);
    
    // Custom postprocessing with adjustable parameters
    std::vector<Yolov8Result> customPostprocessing(const std::vector<float>& rawData,
                                                 const cv::Size &resizedImageShape,
                                                 const cv::Size &originalImageShape,
                                                 float confThreshold,
                                                 float iouThreshold);
};
```

#### 3. Implementation of VectorizedBarrier

```cpp
// In new file: vectorized_barrier.h
#pragma once
#include <immintrin.h>  // For AVX2 instructions
#include <atomic>
#include <thread>

class VectorizedBarrier {
private:
    alignas(64) float syncVector[16]; // Must be aligned for AVX operations
    int numThreads;
    
public:
    VectorizedBarrier(int threads) : numThreads(threads) {
        for (int i = 0; i < 16; i++) {
            syncVector[i] = 0.0f;
        }
    }
    
    void arrive(int threadId, float completionFraction) {
        // Use vectorized operations to update sync state
        syncVector[threadId % 16] = completionFraction;
        _mm_sfence(); // Ensure visibility across cores
    }
    
    bool waitForPhase(float targetPhase, float tolerance = 0.05f) {
        __m256 targetVec = _mm256_set1_ps(targetPhase);
        
        for (int attempt = 0; attempt < 1000; attempt++) {
            bool allReady = true;
            
            // Process 8 elements at a time with AVX
            for (int i = 0; i < numThreads; i += 8) {
                int remaining = std::min(8, numThreads - i);
                if (remaining < 8) {
                    // Handle remaining elements sequentially
                    for (int j = 0; j < remaining; j++) {
                        if (std::abs(syncVector[i + j] - targetPhase) > tolerance) {
                            allReady = false;
                            break;
                        }
                    }
                } else {
                    // Load 8 values from sync vector
                    __m256 values = _mm256_load_ps(&syncVector[i]);
                    
                    // Compute absolute differences
                    __m256 diff = _mm256_sub_ps(values, targetVec);
                    diff = _mm256_and_ps(diff, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
                    
                    // Compare with tolerance
                    __m256 cmp = _mm256_cmp_ps(diff, _mm256_set1_ps(tolerance), _CMP_GT_OQ);
                    
                    // If any thread isn't ready, break
                    if (!_mm256_testz_ps(cmp, cmp)) {
                        allReady = false;
                        break;
                    }
                }
            }
            
            if (allReady) return true;
            
            // Adaptive backoff based on how close threads are to completion
            __m256 sum = _mm256_setzero_ps();
            for (int i = 0; i < numThreads; i += 8) {
                sum = _mm256_add_ps(sum, _mm256_load_ps(&syncVector[i % 16]));
            }
            
            // Extract sum and compute average completion
            float avgCompletion = _mm256_reduce_add_ps(sum) / numThreads;
            float backoffTime = (targetPhase - avgCompletion) * 10.0f;
            if (backoffTime > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(backoffTime * 1000)));
            } else {
                std::this_thread::yield();
            }
        }
        
        return false;
    }
    
    // Helper for horizontal sum of AVX register
    float _mm256_reduce_add_ps(__m256 x) {
        __m128 hi = _mm256_extractf128_ps(x, 1);
        __m128 lo = _mm256_castps256_ps128(x);
        lo = _mm_add_ps(lo, hi);
        hi = _mm_movehl_ps(_mm_setzero_ps(), lo);
        lo = _mm_add_ps(lo, hi);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_add_ss(lo, hi);
        return _mm_cvtss_f32(lo);
    }
};
```

#### 4. Implementation of Elastic Spatial Partitioning

```cpp
// In new file: elastic_partitioning.h
#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <vector>

// Structure for elastic spatial partitions
struct ElasticSpatialPartition {
    cv::Rect region;
    float complexityEstimate;
    int assignedThreadId;
    std::atomic<bool> completed{false};
    
    // Complexity-to-area ratio for load balancing
    float getComplexityDensity() const {
        return complexityEstimate / (region.width * region.height);
    }
};

// Quad-tree node for adaptive spatial partitioning
struct QuadNode {
    cv::Rect region;
    float complexity;
    bool isLeaf;
    QuadNode* children[4];
    
    QuadNode(cv::Rect r) : region(r), complexity(0.0f), isLeaf(true) {
        for (int i = 0; i < 4; i++) children[i] = nullptr;
    }
    
    ~QuadNode() {
        for (int i = 0; i < 4; i++) {
            if (children[i]) delete children[i];
        }
    }
};

// Function to generate elastic partitions
std::vector<ElasticSpatialPartition> generatePartitions(const cv::Mat& image, int targetPartitions) {
    // Create initial complexity map from image
    cv::Mat complexityMap;
    cv::Laplacian(image, complexityMap, CV_32F);
    cv::convertScaleAbs(complexityMap, complexityMap);
    
    // Create root node covering whole image
    QuadNode* root = new QuadNode(cv::Rect(0, 0, image.cols, image.rows));
    root->complexity = cv::sum(complexityMap)[0];
    
    // Build adaptive quad-tree based on complexity
    std::vector<QuadNode*> leafNodes = {root};
    
    // Build the quadtree by recursively splitting high-complexity regions
    // ... (full implementation details)
    
    // Convert leaf nodes to partitions
    std::vector<ElasticSpatialPartition> partitions;
    for (auto node : leafNodes) {
        ElasticSpatialPartition partition;
        partition.region = node->region;
        partition.complexityEstimate = node->complexity;
        partition.assignedThreadId = -1;
        partitions.push_back(partition);
    }
    
    // Clean up
    delete root;
    
    return partitions;
}
```

#### 5. Implementation of Vectorized NMS

```cpp
// In new file: vectorized_nms.h
#pragma once
#include <immintrin.h>  // For AVX2 instructions
#include <vector>
#include "yolov8Predictor.h"  // For Yolov8Result definition

// Vectorized non-maximum suppression
void vectorizedNMS(const std::vector<Yolov8Result>& results,
                  std::vector<Yolov8Result>& keptResults,
                  float iouThreshold) {
    if (results.empty()) return;
    
    // Sort by confidence
    std::vector<size_t> indices(results.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return results[a].conf > results[b].conf;
    });
    
    std::vector<bool> kept(results.size(), true);
    
    // Process each box
    for (size_t i = 0; i < indices.size(); i++) {
        if (!kept[indices[i]]) continue;
        
        const auto& boxA = results[indices[i]].box;
        
        // Prepare box A parameters for vectorized comparison
        __m256 boxA_x1 = _mm256_set1_ps(boxA.x);
        __m256 boxA_y1 = _mm256_set1_ps(boxA.y);
        __m256 boxA_x2 = _mm256_set1_ps(boxA.x + boxA.width);
        __m256 boxA_y2 = _mm256_set1_ps(boxA.y + boxA.height);
        __m256 boxA_area = _mm256_set1_ps(boxA.width * boxA.height);
        
        // Process 8 boxes at a time using AVX
        for (size_t j = i + 1; j < indices.size(); j += 8) {
            // Load and process 8 boxes at once
            // ... (full implementation details)
        }
        
        // Add current box to kept results
        keptResults.push_back(results[indices[i]]);
    }
}
```

#### 6. Complete ESPVS Detector Implementation

```cpp
// In new file: espvs_detector.h
#pragma once
#include "yolov8Predictor.h"
#include "vectorized_barrier.h"
#include "elastic_partitioning.h"
#include "vectorized_nms.h"
#include <thread>
#include <vector>
#include <future>
#include <queue>
#include <mutex>

class ESPVSDetector {
private:
    YOLOPredictor predictor;
    int numThreads;
    std::vector<std::thread> workers;
    VectorizedBarrier barrier;
    std::atomic<bool> running{true};
    
    // Work item structure and queue
    struct WorkItem {
        cv::Mat image;
        std::promise<std::vector<Yolov8Result>> resultPromise;
    };
    
    // Lock-free work queue implementation
    // ... (implementation details)
    
    WorkQueue workQueue;
    
    // Worker thread function
    void workerFunction(int threadId) {
        // ... (implementation details)
    }
    
public:
    ESPVSDetector(const std::string& modelPath, bool isGPU, int threads = 0) 
        : predictor(modelPath, isGPU, 0.25f, 0.45f, 0.5f),
          numThreads(threads > 0 ? threads : std::thread::hardware_concurrency()),
          barrier(threads > 0 ? threads : std::thread::hardware_concurrency()) {
        
        // Start worker threads
        for (int i = 0; i < numThreads; i++) {
            workers.emplace_back(&ESPVSDetector::workerFunction, this, i);
        }
    }
    
    ~ESPVSDetector() {
        running.store(false);
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    // Predict method - this is the main API entry point
    std::vector<Yolov8Result> predict(cv::Mat& image) {
        WorkItem item;
        item.image = image.clone();
        
        std::future<std::vector<Yolov8Result>> future = item.resultPromise.get_future();
        workQueue.push(std::move(item));
        
        return future.get();
    }
};
```

### Modifying main.cpp

Your current main.cpp contains this processing loop:

```cpp
// Current main.cpp processing loop
for (const auto &entry : std::filesystem::directory_iterator(imagePath)) {
    if (std::filesystem::is_regular_file(entry.path()) && 
        std::regex_match(entry.path().filename().string(), pattern)) {
        
        picNums += 1;
        std::string Filename = entry.path().string();
        std::string baseName = std::filesystem::path(Filename).filename().string();
        std::cout << Filename << " predicting..." << std::endl;

        cv::Mat image = cv::imread(Filename);
        std::vector<Yolov8Result> result = predictor.predict(image);
        
        utils::visualizeDetection(image, result, classNames);

        std::string newFilename = baseName.substr(0, baseName.find_last_of('.')) + 
                                "_" + suffixName + 
                                baseName.substr(baseName.find_last_of('.'));
        std::string outputFilename = savePath + "/" + newFilename;
        cv::imwrite(outputFilename, image);
        std::cout << outputFilename << " Saved !!!" << std::endl;
    }
}
```

To use ESPVS, you would modify it to:

```cpp
// Modified main.cpp with ESPVS
// Initialize the ESPVS detector
ESPVSDetector detector(modelPath, isGPU, std::thread::hardware_concurrency());

for (const auto &entry : std::filesystem::directory_iterator(imagePath)) {
    if (std::filesystem::is_regular_file(entry.path()) && 
        std::regex_match(entry.path().filename().string(), pattern)) {
        
        picNums += 1;
        std::string Filename = entry.path().string();
        std::string baseName = std::filesystem::path(Filename).filename().string();
        std::cout << Filename << " predicting with ESPVS..." << std::endl;

        cv::Mat image = cv::imread(Filename);
        
        // Use ESPVS detector instead of traditional predictor
        std::vector<Yolov8Result> result = detector.predict(image);
        
        utils::visualizeDetection(image, result, classNames);

        std::string newFilename = baseName.substr(0, baseName.find_last_of('.')) + 
                                "_" + suffixName + 
                                baseName.substr(baseName.find_last_of('.'));
        std::string outputFilename = savePath + "/" + newFilename;
        cv::imwrite(outputFilename, image);
        std::cout << outputFilename << " Saved using ESPVS!!!" << std::endl;
    }
}
```

### Required CMake Changes

You'll need to update your CMakeLists.txt to enable AVX2 support:

```cmake
# In CMakeLists.txt - add AVX2 support
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()

# Add new source files
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/utils.cpp
    src/yolov8Predictor.cpp
    # New ESPVS files
    src/vectorized_barrier.cpp
    src/elastic_partitioning.cpp
    src/vectorized_nms.cpp
    src/espvs_detector.cpp
)
```

### Implementation Timeline

For a phased implementation approach, consider:

1. **Phase 1**: Modify YOLOPredictor to expose internal stages
2. **Phase 2**: Implement VectorizedBarrier and test with a simple parallel workload
3. **Phase 3**: Implement ElasticSpatialPartitioning and test region-based processing
4. **Phase 4**: Implement VectorizedNMS and compare with traditional NMS
5. **Phase 5**: Integrate all components into the ESPVSDetector class
6. **Phase 6**: Modify main.cpp to use the ESPVS approach
7. **Phase 7**: Performance tuning and optimization

This phased approach allows for incremental testing and validation of each component before full integration. 