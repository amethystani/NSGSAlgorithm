#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "yolov8Predictor.h" // Including for result structure reuse
#include "NeuronNode.h"
#include "SpikeQueue.h"
#include <future>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>
#include <atomic>

// Forward declarations
class GraphPartition;
struct PipelineStage;

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
    std::vector<Ort::Value> outputTensors;  // Store model output tensors for processing

    // NSGS specific properties
    std::shared_ptr<SpikeQueue> spikeQueue;
    std::vector<std::shared_ptr<NeuronNode>> graphNodes;
    float globalThresholdMultiplier; // For thermal/power adaptation
    
    // Graph partitioning for data parallelism
    std::vector<std::unique_ptr<GraphPartition>> partitions;
    int numPartitions;
    
    // Pipeline parallelism
    bool usePipeline;
    std::vector<std::unique_ptr<PipelineStage>> pipelineStages;
    std::mutex pipelineMutex;
    std::condition_variable pipelineCondition;
    std::atomic<bool> pipelineRunning;
    std::thread pipelineThread;
    std::queue<cv::Mat> inputQueue;
    std::queue<std::vector<Yolov8Result>> outputQueue;
    
    // Partitioning and parallel execution methods
    void createGraphPartitions();
    void syncPartitionBoundaries();
    void startPipeline();
    void stopPipeline();
    void pipelineWorker();
    
    // Helper methods
    void getBestClassInfo(std::vector<float>::iterator it, float &bestConf, int &bestClassId, const int _classNums);
    cv::Mat getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos);
    void buildSegmentationGraph(const cv::Mat &image);
    void runAsyncEventProcessing();
    
    // Event-driven processing methods
    void initializeNodePotentials(const cv::Mat &embeddings);
    void extractNodeFeatures(const cv::Mat &image, const cv::Mat &embeddings);
    void extractNodeFeaturesHierarchical(const cv::Mat &image, const cv::Mat &embeddings);
    void createSelectiveConnections(const cv::Mat &image, const cv::Mat &labels, const std::vector<cv::Point2d> &centers);
    void registerNodeEdgeStrengths(const cv::Mat &image);
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
        classNums(80), // Default to 80 COCO classes
        numPartitions(1),
        usePipeline(false),
        pipelineRunning(false)
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
    
    // Asynchronous prediction for pipeline
    void predictAsync(cv::Mat &image);
    bool getNextPredictionResult(std::vector<Yolov8Result> &results, int timeoutMs = 100);
    
    // NSGS specific preprocessing and postprocessing
    void preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    std::vector<Yolov8Result> postprocessing(const cv::Size &resizedImageShape,
                                           const cv::Size &originalImageShape,
                                           std::vector<Ort::Value> &outputTensors);
    
    // Thermal and power adaptation methods
    void setThermalState(float temperature);
    void adaptsGlobalThreshold(float systemLoad);
    
    // Pipeline control
    void enablePipeline(bool enable) { usePipeline = enable; }
    void setNumPartitions(int num) { numPartitions = std::max(1, num); }
    
    // Direct detection method with better timeout handling
    std::vector<Yolov8Result> detect(cv::Mat& image, float confThreshold, float iouThreshold, float maskThreshold);
}; 

// Graph partition class for data parallelism
class GraphPartition {
private:
    int partitionId;
    std::vector<std::shared_ptr<NeuronNode>> nodes;
    std::vector<int> boundaryNodeIndices;
    std::vector<std::pair<int, int>> externalConnections; // (local node index, remote node global index)
    std::shared_ptr<SpikeQueue> localSpikeQueue;
    std::thread processingThread;
    std::atomic<bool> active;
    std::mutex boundaryMutex;
    
    // Reference to the parent predictor's global node array
    std::vector<std::shared_ptr<NeuronNode>>* globalNodesRef;
    
    // Process nodes in this partition
    void processPartition();
    
public:
    GraphPartition(int id, std::vector<std::shared_ptr<NeuronNode>>* globalNodes);
    ~GraphPartition();
    
    // Add a node to this partition
    void addNode(std::shared_ptr<NeuronNode> node, bool isBoundary = false);
    
    // Add an external connection (to node in another partition)
    void addExternalConnection(int localNodeIndex, int remoteNodeGlobalIndex);
    
    // Synchronization with other partitions
    void syncBoundaryNodes();
    
    // Activation
    void start();
    void stop();
    
    // Getters
    int getId() const { return partitionId; }
    size_t getNodeCount() const { return nodes.size(); }
    size_t getBoundaryNodeCount() const { return boundaryNodeIndices.size(); }
    const std::vector<std::shared_ptr<NeuronNode>>& getNodes() const { return nodes; }
    const std::vector<int>& getBoundaryNodeIndices() const { return boundaryNodeIndices; }
};

// Structure for pipeline stages
struct PipelineStage {
    enum class StageType {
        INPUT_PREPROCESSING,
        MODEL_INFERENCE,
        GRAPH_CONSTRUCTION,
        SPIKE_PROPAGATION,
        RESULT_GENERATION
    };
    
    StageType type;
    std::function<void()> process;
    std::thread worker;
    std::atomic<bool> active;
    std::mutex stageMutex;
    std::condition_variable stageCondition;
    
    // Input/output queues
    std::queue<cv::Mat> inputImages;
    std::queue<std::vector<float>> preprocessedData;
    std::queue<std::vector<Ort::Value>> modelOutputs;
    std::queue<std::vector<std::shared_ptr<NeuronNode>>> graphData;
    std::queue<std::vector<Yolov8Result>> results;
    
    PipelineStage(StageType t) : type(t), active(false) {}
    
    ~PipelineStage() {
        active.store(false);
        stageCondition.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    void start(std::function<void()> processFunc) {
        active.store(true);
        process = processFunc;
        worker = std::thread([this]() {
            while (active.load()) {
                {
                    std::unique_lock<std::mutex> lock(stageMutex);
                    stageCondition.wait(lock, [this]() {
                        return !active.load() || 
                               (!inputImages.empty() || 
                                !preprocessedData.empty() || 
                                !modelOutputs.empty() || 
                                !graphData.empty());
                    });
                }
                
                if (active.load() && process) {
                    process();
                }
            }
        });
    }
    
    void stop() {
        active.store(false);
        stageCondition.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }
}; 