#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "yolov8Predictor.h" // Including for result structure reuse
#include "NeuronNode.h"
#include "SpikeQueue.h"

// NSGS: Neuro-Scheduling for Graph Segmentation
class NsgsPredictor {
private:
    // Model and session properties
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    
    // Model configuration
    float confThreshold;
    float iouThreshold;
    float maskThreshold;
    bool hasMask;
    bool isDynamicInputShape;

    // ONNX input and output information
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;

    // NSGS specific properties
    std::shared_ptr<SpikeQueue> spikeQueue;
    std::vector<std::shared_ptr<NeuronNode>> graphNodes;
    float globalThresholdMultiplier; // For thermal/power adaptation
    
    // Helper methods
    void getBestClassInfo(std::vector<float>::iterator it, float &bestConf, int &bestClassId, const int _classNums);
    cv::Mat getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos);
    void buildSegmentationGraph(const cv::Mat &image);
    void runAsyncEventProcessing();
    
    // Event-driven processing methods
    void initializeNodePotentials(const cv::Mat &embeddings);
    void propagateSpikes(bool adaptToThermal = true);
    cv::Mat reconstructFromNeuralGraph();

public:
    int classNums; // Made public for compatibility with YOLOPredictor
    
    // Default constructor
    NsgsPredictor() : 
        env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "NSGS"),
        session(nullptr),
        confThreshold(0.5f),
        iouThreshold(0.5f),
        maskThreshold(0.5f),
        hasMask(false),
        isDynamicInputShape(false),
        globalThresholdMultiplier(1.0f),
        classNums(80) // Default to 80 COCO classes
    {
        // Initialize an empty spike queue
        this->spikeQueue = std::make_shared<SpikeQueue>();
    }
    
    // Main constructor
    NsgsPredictor(const std::string &modelPath,
                 const bool &isGPU,
                 float confThreshold,
                 float iouThreshold,
                 float maskThreshold);
    
    // Main prediction method - follows same interface as YOLOPredictor for compatibility
    std::vector<Yolov8Result> predict(cv::Mat &image);
    
    // NSGS specific preprocessing and postprocessing
    void preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    std::vector<Yolov8Result> postprocessing(const cv::Size &resizedImageShape,
                                           const cv::Size &originalImageShape,
                                           std::vector<Ort::Value> &outputTensors);
    
    // Thermal and power adaptation methods
    void setThermalState(float temperature);
    void adaptsGlobalThreshold(float systemLoad);
}; 