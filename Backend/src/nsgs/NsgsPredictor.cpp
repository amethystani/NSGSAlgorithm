#include "NsgsPredictor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include "utils.h"

NsgsPredictor::NsgsPredictor(const std::string &modelPath,
                             const bool &isGPU,
                             float confThreshold,
                             float iouThreshold,
                             float maskThreshold)
    : env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "NSGS"),
      session(nullptr) // Initialize with nullptr to be set later
{
    this->confThreshold = confThreshold;
    this->iouThreshold = iouThreshold;
    this->maskThreshold = maskThreshold;
    this->globalThresholdMultiplier = 1.0f;
    
    // Initialize ONNX environment
    sessionOptions = Ort::SessionOptions();

    // Check for GPU availability
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    // Initialize session
#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    // Get model information
    const size_t num_input_nodes = session.GetInputCount();
    const size_t num_output_nodes = session.GetOutputCount();
    
    if (num_output_nodes > 1)
    {
        this->hasMask = true;
        std::cout << "NSGS: Instance Segmentation Model" << std::endl;
    }
    else
    {
        std::cout << "NSGS: Object Detection Model" << std::endl;
    }

    // Process input/output names and shapes
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        this->inputNames.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->inputShapes.push_back(inputTensorShape);
        this->isDynamicInputShape = false;
        
        // Check for dynamic shapes
        if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
        {
            std::cout << "Dynamic input shape" << std::endl;
            this->isDynamicInputShape = true;
        }
    }
    
    for (int i = 0; i < num_output_nodes; i++)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        this->outputNames.push_back(output_name.get());
        output_names_ptr.push_back(std::move(output_name));

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        std::vector<int64_t> outputTensorShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->outputShapes.push_back(outputTensorShape);
        
        if (i == 0)
        {
            if (!this->hasMask)
                classNums = outputTensorShape[1] - 4;
            else
                classNums = outputTensorShape[1] - 4 - 32;
        }
    }
    
    // Initialize NSGS specific components
    this->spikeQueue = std::make_shared<SpikeQueue>();
    
    // Set thermal feedback callback
    spikeQueue->setThermalFeedbackCallback([this](float load) {
        this->adaptsGlobalThreshold(load);
    });
    
    std::cout << "NSGS Predictor initialized with " << classNums << " classes" << std::endl;
}

void NsgsPredictor::getBestClassInfo(std::vector<float>::iterator it,
                                    float &bestConf,
                                    int &bestClassId,
                                    const int _classNums)
{
    // First 4 elements are box coordinates
    bestClassId = 4;
    bestConf = 0;

    for (int i = 4; i < _classNums + 4; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 4;
        }
    }
}

cv::Mat NsgsPredictor::getMask(const cv::Mat &maskProposals,
                              const cv::Mat &maskProtos)
{
    cv::Mat protos = maskProtos.reshape(0, {(int)this->outputShapes[1][1], (int)this->outputShapes[1][2] * (int)this->outputShapes[1][3]});

    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks = matmul_res.reshape(1, {(int)this->outputShapes[1][2], (int)this->outputShapes[1][3]});
    cv::Mat dest;

    // Apply sigmoid activation
    cv::exp(-masks, dest);
    dest = 1.0 / (1.0 + dest);
    cv::resize(dest, dest, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]), cv::INTER_LINEAR);
    return dest;
}

void NsgsPredictor::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape)
{
    // Standard preprocessing similar to YOLOPredictor
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]),
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    // hwc -> chw conversion
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
    
    // Additional step: prepare the segmentation graph based on this image
    buildSegmentationGraph(resizedImage);
}

void NsgsPredictor::buildSegmentationGraph(const cv::Mat &image)
{
    // Clear any existing nodes
    graphNodes.clear();
    
    // Create a downsampled grid of nodes
    // For efficiency, we don't create a node for each pixel
    const int gridStep = 8; // Create a node every 8 pixels
    const int width = image.cols;
    const int height = image.rows;
    
    // Create nodes
    int nodeId = 0;
    for (int y = 0; y < height; y += gridStep) {
        for (int x = 0; x < width; x += gridStep) {
            auto node = std::make_shared<NeuronNode>(nodeId++, cv::Point2i(x, y), spikeQueue);
            graphNodes.push_back(node);
        }
    }
    
    std::cout << "NSGS: Created " << graphNodes.size() << " graph nodes" << std::endl;
    
    // Create connections between nodes (8-neighborhood)
    const int rows = height / gridStep;
    const int cols = width / gridStep;
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            auto &node = graphNodes[idx];
            
            // Connect to 8 neighbors
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue; // Skip self
                    
                    int nr = r + dr;
                    int nc = c + dc;
                    
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                        int nidx = nr * cols + nc;
                        float weight = (dr == 0 || dc == 0) ? 1.0f : 0.7071f; // Diagonal connections have lower weight
                        node->addConnection(graphNodes[nidx], weight);
                    }
                }
            }
        }
    }
    
    // Start async spike processing
    spikeQueue->startProcessing(&graphNodes);
}

void NsgsPredictor::initializeNodePotentials(const cv::Mat &embeddings)
{
    // This would be called with the embeddings from the model
    // For now, we'll just initialize with some default potentials based on position
    
    for (auto &node : graphNodes) {
        // Reset node state
        node->resetState();
        
        // Set initial threshold based on position (just as an example)
        cv::Point2i pos = node->getPosition();
        float threshold = 0.5f + 0.2f * sin(pos.x * 0.1f) * cos(pos.y * 0.1f);
        node->setThreshold(threshold);
        
        // Initialize with a small random potential to break symmetry
        float initialPotential = 0.1f * static_cast<float>(rand()) / RAND_MAX;
        node->incrementPotential(initialPotential);
    }
}

void NsgsPredictor::propagateSpikes(bool adaptToThermal)
{
    // This would typically be handled by the SpikeQueue's async processing
    // But we can force some initial spikes for key nodes
    
    // Seed the network with some initial spikes at key positions
    const int numInitialSpikes = 10; // Number of random nodes to activate
    for (int i = 0; i < numInitialSpikes; i++) {
        int nodeIdx = rand() % graphNodes.size();
        graphNodes[nodeIdx]->incrementPotential(1.0f); // Strong initial activation
        graphNodes[nodeIdx]->checkAndFire(); // Check if this causes a spike
    }
    
    // Allow the system to process for a fixed amount of time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Adapt thresholds based on thermal state if requested
    if (adaptToThermal) {
        for (auto &node : graphNodes) {
            node->adaptThreshold(globalThresholdMultiplier);
        }
    }
}

cv::Mat NsgsPredictor::reconstructFromNeuralGraph()
{
    // Create a blank mask image
    int width = (int)this->inputShapes[0][3];
    int height = (int)this->inputShapes[0][2];
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    
    // For each node, color its area based on its class ID
    for (auto &node : graphNodes) {
        int classId = node->getClassId();
        if (classId >= 0) { // If node has been assigned a class
            cv::Point2i pos = node->getPosition();
            int radius = 4; // Radius of influence for each node
            
            // Draw a filled circle at the node's position with its class ID as the value
            cv::circle(mask, pos, radius, cv::Scalar(classId + 1), -1); // +1 to avoid 0 (background)
        }
    }
    
    return mask;
}

void NsgsPredictor::runAsyncEventProcessing()
{
    // This would initiate the event-driven processing loop
    // For now, we'll just simulate it with a sleep
    std::cout << "NSGS: Running asynchronous event processing..." << std::endl;
    
    // Initialize node potentials
    cv::Mat dummyEmbeddings; // In a real implementation, this would be from the model
    initializeNodePotentials(dummyEmbeddings);
    
    // Start spike propagation
    propagateSpikes(true);
    
    // Wait for processing to finish (in a real impl, would use a barrier or callback)
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "NSGS: Processed " << spikeQueue->getProcessedCount() << " spikes" << std::endl;
    std::cout << "NSGS: Max queue size was " << spikeQueue->getHighWatermark() << std::endl;
}

std::vector<Yolov8Result> NsgsPredictor::postprocessing(const cv::Size &resizedImageShape,
                                                       const cv::Size &originalImageShape,
                                                       std::vector<Ort::Value> &outputTensors)
{
    // Similar structure to YOLOPredictor but using NSGS processing
    
    // Box outputs
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float *boxOutput = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
    float *output0ptr = (float *)output0.data;
    int rows = (int)this->outputShapes[0][2];
    int cols = (int)this->outputShapes[0][1];
    
    // For mask handling
    std::vector<std::vector<float>> picked_proposals;
    cv::Mat mask_protos;

    // Process detection boxes
    for (int i = 0; i < rows; i++)
    {
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);

        if (confidence > this->confThreshold)
        {
            if (this->hasMask)
            {
                std::vector<float> temp(it.begin() + 4 + classNums, it.end());
                picked_proposals.push_back(temp);
            }
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);

    // Process masks if available
    if (this->hasMask)
    {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        std::vector<int> mask_protos_shape = {1, (int)this->outputShapes[1][1], (int)this->outputShapes[1][2], (int)this->outputShapes[1][3]};
        mask_protos = cv::Mat(mask_protos_shape, CV_32F, maskOutput);
    }

    // Process neural graph to get instance segmentation masks
    runAsyncEventProcessing();
    cv::Mat neurographMask = reconstructFromNeuralGraph();
    
    // Prepare results
    std::vector<Yolov8Result> results;
    for (int idx : indices)
    {
        Yolov8Result res;
        res.box = cv::Rect(boxes[idx]);
        
        if (this->hasMask)
        {
            if (!picked_proposals.empty()) {
                res.boxMask = this->getMask(cv::Mat(picked_proposals[idx]).t(), mask_protos);
                
                // Here we would incorporate the neurograph mask, but for now we'll just use the original
                // In a full implementation, we would blend or refine the original mask with our neurograph result
            }
            else {
                res.boxMask = neurographMask.clone(); // Use our generated mask
            }
        }
        else
        {
            res.boxMask = cv::Mat::zeros((int)this->inputShapes[0][2], (int)this->inputShapes[0][3], CV_8U);
        }

        utils::scaleCoords(res.box, res.boxMask, this->maskThreshold, resizedImageShape, originalImageShape);
        res.conf = confs[idx];
        res.classId = classIds[idx];
        results.emplace_back(res);
    }

    // Stop spike processing
    spikeQueue->stopProcessing();

    return results;
}

std::vector<Yolov8Result> NsgsPredictor::predict(cv::Mat &image)
{
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

    // Run inference
    std::vector<Ort::Value> outputTensors = this->session.Run(
        Ort::RunOptions{nullptr},
        this->inputNames.data(),
        inputTensors.data(),
        this->inputNames.size(),
        this->outputNames.data(),
        this->outputNames.size());

    // Post-process using NSGS approach
    std::vector<Yolov8Result> result = this->postprocessing(
        cv::Size(inputTensorShape[3], inputTensorShape[2]),
        cv::Size(image.cols, image.rows),
        outputTensors);

    delete[] blob;
    return result;
}

void NsgsPredictor::setThermalState(float temperature)
{
    // Adjust global threshold based on device temperature
    // Higher temperature -> higher threshold -> less activity
    float normalizedTemp = std::min(1.0f, std::max(0.0f, (temperature - 30.0f) / 50.0f));
    globalThresholdMultiplier = 1.0f + normalizedTemp;
    
    std::cout << "NSGS: Thermal state updated. Threshold multiplier = " << globalThresholdMultiplier << std::endl;
}

void NsgsPredictor::adaptsGlobalThreshold(float systemLoad)
{
    // Adjust based on system load (0.0 to 1.0)
    float loadAdjustment = 1.0f + systemLoad;
    globalThresholdMultiplier = std::min(2.0f, loadAdjustment);
    
    std::cout << "NSGS: System load adaptation. New threshold multiplier = " << globalThresholdMultiplier << std::endl;
} 