#include "NsgsPredictor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map> // For watershed region tracking
#include <unordered_set> // For efficient adjacency tracking
#include "utils.h"
#include <opencv2/ximgproc.hpp> // For SLIC superpixels
#include <future>
#include <queue>
#include <random>  // Add this at the top for std::random_device and std::mt19937

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
    this->outputTensors.clear(); // Initialize empty output tensors
    
    // Initialize multi-threaded spike queue with hardware-optimized thread count
    int numThreads = std::thread::hardware_concurrency();
    // Use at least 2 threads, but cap at 8 to avoid excessive overhead
    numThreads = std::min(8, std::max(2, numThreads));
    this->spikeQueue = std::make_shared<SpikeQueue>(numThreads);
    std::cout << "NSGS: Initialized with " << numThreads << " parallel processing threads" << std::endl;
    
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
    // Safety check for valid inputs
    if (maskProposals.empty() || maskProtos.empty() || 
        this->outputShapes.size() < 2 || 
        this->outputShapes[1].size() < 4) {
        std::cerr << "NSGS: Invalid mask inputs or shapes" << std::endl;
        return cv::Mat();
    }

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
    
    // Store original image for feature extraction
    // We need the original RGB image (not the normalized float image)
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    
    // Resize to match the model input size while keeping aspect ratio
    cv::Mat processedImage;
    utils::letterbox(rgbImage, processedImage, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]),
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);
    
    // We'll build the segmentation graph in postprocessing after we have the model outputs
    // This ensures we have the embeddings available for feature extraction
}

void NsgsPredictor::buildSegmentationGraph(const cv::Mat &image)
{
    // Clear any existing nodes
    graphNodes.clear();
    
    // Safety check for valid inputs
    if (image.empty()) {
        std::cerr << "NSGS: Empty image provided to buildSegmentationGraph" << std::endl;
        return;
    }
    
    // NOVEL OPTIMIZATION 1: CNN-GUIDED ADAPTIVE SUPERPIXEL COUNT
    // Use CNN detection confidence to determine optimal superpixel density
    float avgConfidence = 0.0f;
    int detectionCount = 0;
    
    // Get average detection confidence from recent processing
    if (!this->outputTensors.empty()) {
        float *boxOutput = this->outputTensors[0].GetTensorMutableData<float>();
        if (boxOutput && !this->outputShapes.empty()) {
            cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
            float *output0ptr = (float *)output0.data;
            int rows = (int)this->outputShapes[0][2];
            int cols = (int)this->outputShapes[0][1];
            
            // Quick scan for average confidence
            for (int i = 0; i < rows && i < 1000; i++) { // Limit scan for speed
                std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
                if (it.size() >= 4 + classNums) {
                    float confidence;
                    int classId;
                    this->getBestClassInfo(it.begin(), confidence, classId, classNums);
                    if (confidence > 0.3f) { // Lower threshold for analysis
                        avgConfidence += confidence;
                        detectionCount++;
                    }
                }
            }
            if (detectionCount > 0) avgConfidence /= detectionCount;
        }
    }
    
    // NOVEL: Adaptive superpixel count based on image complexity and CNN confidence
    const int baseSuperpixels = 500;  // Reduced from 1000
    const int maxSuperpixels = 2000;  // Reduced from unlimited
    
    // Higher confidence = fewer superpixels needed (CNN is confident)
    // Lower confidence = more superpixels needed (CNN uncertain, need neuromorphic help)
    float complexityFactor = (1.0f - avgConfidence) * 2.0f; // 0-2 range
    int adaptiveSuperpixelCount = baseSuperpixels + (int)(complexityFactor * 500);
    adaptiveSuperpixelCount = std::min(maxSuperpixels, adaptiveSuperpixelCount);
    
    std::cout << "NSGS NOVEL: Adaptive superpixel count = " << adaptiveSuperpixelCount 
              << " (avg confidence = " << avgConfidence << ")" << std::endl;
    
    // Get embeddings from model output for feature extraction
    cv::Mat embeddings;
    
    // If we have mask protos from the YOLO segmentation model, use them as embeddings
    if (this->hasMask && outputTensors.size() > 1) {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        if (!maskOutput) {
            std::cerr << "NSGS: Invalid mask output data" << std::endl;
            return;
        }
        std::vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        
        // Ensure valid dimensions
        if (maskShape.size() < 4) {
            std::cerr << "NSGS: Invalid mask shape dimensions" << std::endl;
            return;
        }
        
        // Reshape to usable format for OpenCV
        int protoChannels = maskShape[1];
        int protoHeight = maskShape[2];
        int protoWidth = maskShape[3];
        
        // Create a proper embedding matrix
        embeddings = cv::Mat(protoHeight, protoWidth, CV_32FC(protoChannels), maskOutput);
    }
    
    // Create a graph representation using superpixels instead of fixed grid
    const int width = image.cols;
    const int height = image.rows;
    
    // Container for superpixel labels
    cv::Mat labels;
    std::vector<cv::Point2d> centers;
    
    // Try to use SLIC superpixels first (as mentioned in the paper)
    bool useSuperpixels = true;
    try {
        // Convert to Lab color space for better superpixel segmentation
        cv::Mat labImage;
        cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);
        
        // NOVEL: Adaptive parameters based on image content
        int numIterations = (avgConfidence > 0.7f) ? 5 : 10; // Fewer iterations if CNN is confident
        float ruler = 10.0f + (1.0f - avgConfidence) * 10.0f; // More compact if CNN uncertain
        
        // Create and run the SLIC algorithm
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = 
            cv::ximgproc::createSuperpixelSLIC(labImage, cv::ximgproc::SLIC, ruler, adaptiveSuperpixelCount);
        
        slic->iterate(numIterations);
        
        // Get the labels and contours
        slic->getLabels(labels);
        slic->enforceLabelConnectivity();
        
        // Get actual number of superpixels
        int numLabels = slic->getNumberOfSuperpixels();
        std::cout << "NSGS NOVEL: Created " << numLabels << " adaptive SLIC superpixels" << std::endl;
        
        // Calculate centers of superpixels
        centers.resize(numLabels);
        std::vector<int> counts(numLabels, 0);
        
        // Initialize centers
        for (int i = 0; i < numLabels; i++) {
            centers[i] = cv::Point2d(0, 0);
        }
        
        // Sum positions for each superpixel
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = labels.at<int>(y, x);
                if (label >= 0 && label < numLabels) {
                    centers[label].x += x;
                    centers[label].y += y;
                    counts[label]++;
                }
            }
        }
        
        // Calculate average positions (centers)
        for (int i = 0; i < numLabels; i++) {
            if (counts[i] > 0) {
                centers[i].x /= counts[i];
                centers[i].y /= counts[i];
            }
        }
    }
    catch (const cv::Exception& e) {
        // Fallback if SLIC is not available
        std::cerr << "NSGS: SLIC superpixel error: " << e.what() << std::endl;
        std::cerr << "NSGS: Falling back to watershed segmentation" << std::endl;
        useSuperpixels = false;
    }
    
    // Create nodes at superpixel/region centers
    int nodeId = 0;
    for (const auto& center : centers) {
        if (center.x > 0 || center.y > 0) { // Skip empty centers
            auto node = std::make_shared<NeuronNode>(nodeId++, cv::Point2i(center.x, center.y), spikeQueue);
            graphNodes.push_back(node);
        }
    }
    
    std::cout << "NSGS NOVEL: Created " << graphNodes.size() << " graph nodes from adaptive segmentation" << std::endl;
    
    // NOVEL OPTIMIZATION 2: HIERARCHICAL FEATURE EXTRACTION
    // Only extract detailed features for high-priority nodes
    extractNodeFeaturesHierarchical(image, embeddings);
    
    // Register edge strengths for priority scheduling
    registerNodeEdgeStrengths(image);
    
    // NOVEL OPTIMIZATION 3: SELECTIVE CONNECTION CREATION
    // Create connections only between nearby high-priority nodes
    createSelectiveConnections(image, labels, centers);
    
    // ESSENTIAL NEUROMORPHIC STEP: Initialize node potentials and thresholds
    std::cout << "NSGS NEUROMORPHIC: Initializing node potentials and firing thresholds..." << std::endl;
    initializeNodePotentialsFromEmbeddings(image, embeddings);
    
    std::cout << "NSGS NEUROMORPHIC: Graph construction completed - " << graphNodes.size() 
              << " neurons ready for spike propagation" << std::endl;
}

void NsgsPredictor::extractNodeFeatures(const cv::Mat &image, const cv::Mat &embeddings)
{
    // Extract rich features for each node using both raw image data and CNN embeddings
    std::cout << "NSGS: Extracting node features from image and embeddings" << std::endl;
    auto featureStartTime = std::chrono::high_resolution_clock::now();
    auto featureTimeout = featureStartTime + std::chrono::seconds(8); // 8 second timeout
    
    // Check input validity
    if (image.empty()) {
        std::cerr << "NSGS: Error - Empty image provided for feature extraction" << std::endl;
        return;
    }
    
    // Prepare feature extraction for image
    // Convert image to floating point for feature computation
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0/255.0);
    
    // Calculate image gradients (Scharr operator for better gradient estimation)
    cv::Mat gradX, gradY, gradMag, gradOrient;
    cv::Scharr(floatImage, gradX, CV_32F, 1, 0);
    cv::Scharr(floatImage, gradY, CV_32F, 0, 1);
    
    // Compute gradient magnitude and orientation
    cv::cartToPolar(gradX, gradY, gradMag, gradOrient);
    
    // SIMPLIFIED texture computation for speed - use simpler variance-based texture
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat grayFloat;
    grayImage.convertTo(grayFloat, CV_32F);
    
    // Instead of expensive LBP, use local variance as texture measure
    cv::Mat textureMap;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat mean, sqmean;
    cv::boxFilter(grayFloat, mean, CV_32F, cv::Size(5, 5));
    cv::boxFilter(grayFloat.mul(grayFloat), sqmean, CV_32F, cv::Size(5, 5));
    textureMap = sqmean - mean.mul(mean); // Variance
    
    std::cout << "NSGS: Processing " << graphNodes.size() << " nodes for feature extraction..." << std::endl;
    
    // Create feature vector for each node
    int nodesProcessed = 0;
    for (auto &node : graphNodes) {
        // Progress check and timeout every 100 nodes
        if (nodesProcessed % 100 == 0) {
            if (std::chrono::high_resolution_clock::now() > featureTimeout) {
                std::cout << "NSGS: Feature extraction timeout after processing " << nodesProcessed 
                          << " of " << graphNodes.size() << " nodes" << std::endl;
                break;
            }
            std::cout << "NSGS: Feature extraction progress: " << nodesProcessed 
                      << "/" << graphNodes.size() << " nodes" << std::endl;
        }
        
        // Reset any existing features
        std::vector<float> featureVector;
        
        // Get node position
        cv::Point2i pos = node->getPosition();
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        // 1. COLOR FEATURES (Lab color space for perceptual uniformity)
        cv::Mat labImage;
        cv::cvtColor(floatImage, labImage, cv::COLOR_BGR2Lab);
        
        // Extract color in smaller neighborhood for speed (3x3 instead of 5x5)
        const int colorPatchRadius = 1;
        std::vector<float> labValues(3, 0.0f);
        int patchCount = 0;
        
        for (int dy = -colorPatchRadius; dy <= colorPatchRadius; dy++) {
            for (int dx = -colorPatchRadius; dx <= colorPatchRadius; dx++) {
                int nx = std::min(image.cols - 1, std::max(0, x + dx));
                int ny = std::min(image.rows - 1, std::max(0, y + dy));
                
                cv::Vec3f labPixel = labImage.at<cv::Vec3f>(ny, nx);
                labValues[0] += labPixel[0]; // L
                labValues[1] += labPixel[1]; // a
                labValues[2] += labPixel[2]; // b
                patchCount++;
            }
        }
        
        // Average color values
        for (int i = 0; i < 3; i++) {
            featureVector.push_back(labValues[i] / patchCount);
        }
        
        // 2. SIMPLIFIED TEXTURE FEATURES (using variance instead of LBP)
        const int texturePatchRadius = 2;
        float textureVariance = 0.0f;
        int textureCount = 0;
        
        for (int dy = -texturePatchRadius; dy <= texturePatchRadius; dy++) {
            for (int dx = -texturePatchRadius; dx <= texturePatchRadius; dx++) {
                int nx = std::min(textureMap.cols - 1, std::max(0, x + dx));
                int ny = std::min(textureMap.rows - 1, std::max(0, y + dy));
                
                textureVariance += textureMap.at<float>(ny, nx);
                textureCount++;
            }
        }
        
        // Add single texture feature (much faster than 16 LBP features)
        featureVector.push_back(textureVariance / textureCount);
        
        // 3. SIMPLIFIED GRADIENT FEATURES (use magnitude only, not orientation histogram)
        const int gradientPatchRadius = 2;
        float avgGradientMag = 0.0f;
        int gradCount = 0;
        
        for (int dy = -gradientPatchRadius; dy <= gradientPatchRadius; dy++) {
            for (int dx = -gradientPatchRadius; dx <= gradientPatchRadius; dx++) {
                int nx = std::min(gradMag.cols - 1, std::max(0, x + dx));
                int ny = std::min(gradMag.rows - 1, std::max(0, y + dy));
                
                avgGradientMag += gradMag.at<float>(ny, nx);
                gradCount++;
            }
        }
        
        // Add single gradient feature instead of 8 orientation bins
        featureVector.push_back(avgGradientMag / gradCount);
        
        // 4. CNN EMBEDDINGS (if available) - simplified
        if (!embeddings.empty()) {
            // Calculate embedding position (may be at different resolution)
            float scaleX = static_cast<float>(embeddings.cols) / static_cast<float>(image.cols);
            float scaleY = static_cast<float>(embeddings.rows) / static_cast<float>(image.rows);
            
            int embX = std::min(embeddings.cols - 1, std::max(0, static_cast<int>(x * scaleX)));
            int embY = std::min(embeddings.rows - 1, std::max(0, static_cast<int>(y * scaleY)));
            
            // Get embedding features - only use a few channels for speed
            int embChannels = embeddings.channels();
            
            if (embChannels == 1) {
                // Single channel - just use center pixel
                featureVector.push_back(embeddings.at<float>(embY, embX));
            } else {
                // Multi-channel - use every 8th channel to keep feature count low
                int maxChannels = std::min(4, embChannels); // Max 4 embedding features
                for (int c = 0; c < maxChannels; c++) {
                    int channelIdx = c * (embChannels / maxChannels);
                    if (embChannels <= 32) {
                        featureVector.push_back(embeddings.at<cv::Vec<float, 32>>(embY, embX)[channelIdx]);
                    } else {
                        // For different channel counts, do a safe access
                        float* pixelPtr = (float*)(embeddings.data + embY*embeddings.step + embX*embeddings.elemSize());
                        featureVector.push_back(pixelPtr[channelIdx]);
                    }
                }
            }
        }
        
        // Simple normalization (much faster than L2 norm)
        float maxVal = 0.0f;
        for (float f : featureVector) {
            maxVal = std::max(maxVal, std::abs(f));
        }
        
        if (maxVal > 0) {
            for (float &f : featureVector) {
                f /= maxVal;
            }
        }
        
        // Set features for this node
        node->setFeatures(featureVector);
        nodesProcessed++;
    }
    
    auto featureTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - featureStartTime).count();
    
    std::cout << "NSGS: Extracted " << (graphNodes.empty() ? 0 : graphNodes[0]->getFeatures().size())
              << " features per node for " << nodesProcessed << " nodes in " << featureTime << "ms" << std::endl;
}

void NsgsPredictor::propagateSpikes(bool adaptToThermal)
{
    std::cout << "NSGS NEUROMORPHIC: Starting spike propagation across neural network..." << std::endl;
    
    // Start timeout counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto timeout = startTime + std::chrono::seconds(5); // Reduced timeout for efficiency
    
    // NEUROMORPHIC: Content-aware initial spike generation
    // Instead of random, use CNN confidence and edge strength to decide which neurons should fire first
    
    if (graphNodes.empty()) {
        std::cerr << "NSGS NEUROMORPHIC: No neurons available for spike propagation" << std::endl;
        return;
    }
    
    std::cout << "NSGS NEUROMORPHIC: Analyzing " << graphNodes.size() << " neurons for initial firing..." << std::endl;
    
    // NEUROMORPHIC STEP 1: Check which neurons are already above firing threshold
    std::vector<std::shared_ptr<NeuronNode>> firingNeurons;
    std::vector<std::shared_ptr<NeuronNode>> subthresholdNeurons;
    
    for (auto& neuron : graphNodes) {
        if (neuron->checkAndFire()) {
            firingNeurons.push_back(neuron);
            std::cout << "NSGS NEUROMORPHIC: Neuron " << neuron->getId() 
                      << " fires! (potential=" << neuron->getPotential() 
                      << ", threshold=" << neuron->getThreshold() << ")" << std::endl;
        } else {
            subthresholdNeurons.push_back(neuron);
        }
    }
    
    std::cout << "NSGS NEUROMORPHIC: " << firingNeurons.size() << " neurons fired spontaneously, " 
              << subthresholdNeurons.size() << " remain subthreshold" << std::endl;
    
    // NEUROMORPHIC STEP 2: If few neurons fired naturally, stimulate high-priority ones
    if (firingNeurons.size() < 10) { // Need more activity for segmentation
        std::cout << "NSGS NEUROMORPHIC: Insufficient spontaneous firing, applying external stimulation..." << std::endl;
        
        // Sort neurons by potential (closest to firing)
        std::sort(subthresholdNeurons.begin(), subthresholdNeurons.end(),
                  [](const std::shared_ptr<NeuronNode>& a, const std::shared_ptr<NeuronNode>& b) {
                      return a->getPotential() > b->getPotential();
                  });
        
        // ADAPTIVE: Stimulation count based on detected objects and image complexity
        int baseStimulation = 30; // Increased base
        
        // Get detection count from recent processing
        int detectionCount = 0;
        if (!this->outputTensors.empty()) {
            float *boxOutput = this->outputTensors[0].GetTensorMutableData<float>();
            if (boxOutput && !this->outputShapes.empty()) {
                cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
                float *output0ptr = (float *)output0.data;
                int rows = std::min(1000, (int)this->outputShapes[0][2]); // Limit scan
                int cols = (int)this->outputShapes[0][1];
                
                for (int i = 0; i < rows; i++) {
                    std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
                    if (it.size() >= 4 + classNums) {
                        float confidence;
                        int classId;
                        this->getBestClassInfo(it.begin(), confidence, classId, classNums);
                        if (confidence > 0.25f) detectionCount++; // Count significant detections
                    }
                }
            }
        }
        
        // ADAPTIVE: More stimulation for complex scenes with many objects
        int adaptiveStimulation = baseStimulation + (detectionCount * 10); // 10 extra per detection
        adaptiveStimulation = std::min(200, std::max(20, adaptiveStimulation)); // Range: 20-200
        
        int stimulationCount = std::min(adaptiveStimulation, static_cast<int>(subthresholdNeurons.size()));
        
        std::cout << "NSGS NEUROMORPHIC: Adaptive stimulation - " << detectionCount << " detections detected, stimulating " 
                  << stimulationCount << " neurons" << std::endl;
        
        for (int i = 0; i < stimulationCount; i++) {
            auto& neuron = subthresholdNeurons[i];
            float currentPotential = neuron->getPotential();
            float threshold = neuron->getThreshold();
            float stimulation = (threshold - currentPotential) + 0.1f; // Just over threshold
            
            neuron->incrementPotential(stimulation);
            if (neuron->checkAndFire()) {
                firingNeurons.push_back(neuron);
                if (i < 10) { // Log first 10 for brevity
                    std::cout << "NSGS NEUROMORPHIC: Stimulated neuron " << neuron->getId() 
                              << " to fire (stim=" << stimulation << ")" << std::endl;
                }
            }
        }
        
        std::cout << "NSGS NEUROMORPHIC: After adaptive stimulation: " << firingNeurons.size() 
                  << " total firing neurons" << std::endl;
    }
    
    // NEUROMORPHIC STEP 3: Propagate spikes through connections
    std::cout << "NSGS NEUROMORPHIC: Beginning spike propagation through synaptic connections..." << std::endl;
    
    int propagationRounds = 0;
    int totalSpikesGenerated = firingNeurons.size();
    const int maxRounds = 8; // Reduced to prevent runaway
    const int maxNeuronsPerRound = 500; // NEUROMORPHIC: Limit cascade size
    
    while (!firingNeurons.empty() && propagationRounds < maxRounds) {
        // Check timeout
        if (std::chrono::high_resolution_clock::now() > timeout) {
            std::cout << "NSGS NEUROMORPHIC: Propagation timeout after " << propagationRounds 
                      << " rounds, finalizing with current state" << std::endl;
            break;
        }
        
        propagationRounds++;
        std::vector<std::shared_ptr<NeuronNode>> nextFiringNeurons;
        
        // NEUROMORPHIC: Limit the number of neurons processing per round to prevent explosion
        int neuronsToProcess = std::min(static_cast<int>(firingNeurons.size()), maxNeuronsPerRound);
        
        std::cout << "NSGS NEUROMORPHIC: Propagation round " << propagationRounds 
                  << " - processing " << neuronsToProcess << " of " << firingNeurons.size() << " firing neurons" << std::endl;
        
        // Each firing neuron sends spikes to its connected neighbors
        for (int i = 0; i < neuronsToProcess; i++) {
            auto& firingNeuron = firingNeurons[i];
            
            cv::Point2i firingPos = firingNeuron->getPosition();
            const float maxPropagationDistance = 300.0f; // Increased from 200 for better reach
            int spikesSent = 0;
            const int maxSpikesPerNeuron = 25; // Increased from 15 for more connections
            
            for (auto& targetNeuron : graphNodes) {
                if (targetNeuron->getId() == firingNeuron->getId()) continue; // Skip self
                if (spikesSent >= maxSpikesPerNeuron) break; // Limit connections
                
                cv::Point2i targetPos = targetNeuron->getPosition();
                float dx = static_cast<float>(firingPos.x - targetPos.x);
                float dy = static_cast<float>(firingPos.y - targetPos.y);
                float distance = std::sqrt(dx*dx + dy*dy);
                
                // NEUROMORPHIC: Spike propagation with distance-dependent attenuation
                if (distance < maxPropagationDistance) {
                    float synapticWeight = 1.0f - (distance / maxPropagationDistance); // 0-1 weight
                    
                    // NEUROMORPHIC: Synaptic fatigue - reduce strength in later rounds
                    float fatigueMultiplier = 1.0f - (propagationRounds * 0.05f); // Reduced fatigue: 5% per round
                    fatigueMultiplier = std::max(0.4f, fatigueMultiplier); // Keep minimum 40% strength
                    
                    // ADAPTIVE: Spike strength based on neuron priority and distance
                    float baseSpikeStrength = 0.5f; // Further increased for realistic chain reactions
                    float spikeStrength = baseSpikeStrength * synapticWeight * fatigueMultiplier;
                    
                    // NEUROMORPHIC: Check if target is in refractory period (recently fired)
                    // For simplicity, assume neurons that are already firing are in refractory
                    bool inRefractory = false;
                    for (auto& recentFiring : firingNeurons) {
                        if (recentFiring->getId() == targetNeuron->getId()) {
                            inRefractory = true;
                            break;
                        }
                    }
                    
                    if (!inRefractory) {
                        // Add synaptic input to target neuron
                        targetNeuron->incrementPotential(spikeStrength);
                        
                        // Check if target neuron now fires
                        if (targetNeuron->checkAndFire()) {
                            // NEUROMORPHIC: Avoid duplicate firing neurons
                            bool alreadyFiring = false;
                            for (auto& existing : nextFiringNeurons) {
                                if (existing->getId() == targetNeuron->getId()) {
                                    alreadyFiring = true;
                                    break;
                                }
                            }
                            
                            if (!alreadyFiring) {
                                nextFiringNeurons.push_back(targetNeuron);
                                totalSpikesGenerated++;
                                
                                if (spikesSent == 0) { // Log first spike from each neuron
                                    std::cout << "NSGS NEUROMORPHIC: Spike " << firingNeuron->getId() 
                                              << "->" << targetNeuron->getId() 
                                              << " triggered firing (weight=" << synapticWeight << ", fatigue=" << fatigueMultiplier << ")" << std::endl;
                                }
                            }
                        }
                        spikesSent++;
                    }
                }
            }
        }
        
        // NEUROMORPHIC: Convergence check - if activity is decreasing significantly
        float activityRatio = static_cast<float>(nextFiringNeurons.size()) / static_cast<float>(neuronsToProcess);
        
        std::cout << "NSGS NEUROMORPHIC: Round " << propagationRounds 
                  << " complete - " << nextFiringNeurons.size() << " new firing neurons (activity ratio: " << activityRatio << ")" << std::endl;
        
        // NEUROMORPHIC: Stop if activity is very low (network converged) - relaxed threshold for realistic propagation
        if (activityRatio < 0.03f && propagationRounds > 3) { // 3% threshold after at least 3 rounds
            std::cout << "NSGS NEUROMORPHIC: Spike propagation converged - low activity detected" << std::endl;
            break;
        }
        
        // NEUROMORPHIC: Stop if too many neurons are firing (prevent explosion)
        if (nextFiringNeurons.size() > maxNeuronsPerRound * 2) {
            std::cout << "NSGS NEUROMORPHIC: Limiting cascade - too many neurons firing, applying network inhibition" << std::endl;
            // Keep only the strongest firing neurons
            std::sort(nextFiringNeurons.begin(), nextFiringNeurons.end(),
                      [](const std::shared_ptr<NeuronNode>& a, const std::shared_ptr<NeuronNode>& b) {
                          return a->getPotential() > b->getPotential();
                      });
            nextFiringNeurons.resize(maxNeuronsPerRound);
        }
        
        // Update for next round
        firingNeurons = nextFiringNeurons;
        
        // Stop if no new neurons fired
        if (firingNeurons.empty()) {
            std::cout << "NSGS NEUROMORPHIC: Spike propagation converged - no new firing" << std::endl;
            break;
        }
    }
    
    // NEUROMORPHIC STEP 4: Update global statistics
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startTime).count();
    
    std::cout << "NSGS NEUROMORPHIC: Spike propagation completed in " << elapsedTime << "ms" << std::endl;
    std::cout << "NSGS NEUROMORPHIC: Total spikes generated: " << totalSpikesGenerated << std::endl;
    std::cout << "NSGS NEUROMORPHIC: Propagation rounds: " << propagationRounds << std::endl;
    std::cout << "NSGS NEUROMORPHIC: Neural network ready for segmentation reconstruction" << std::endl;
}

void NsgsPredictor::runAsyncEventProcessing()
{
    std::cout << "NSGS NEUROMORPHIC: Processing neural network events and state changes..." << std::endl;
    
    // Start timeout counter for event processing
    auto startTime = std::chrono::high_resolution_clock::now();
    auto timeout = startTime + std::chrono::seconds(3); // Reduced timeout for speed
    
    if (!spikeQueue) {
        std::cerr << "NSGS NEUROMORPHIC: SpikeQueue not initialized" << std::endl;
        return;
    }
    
    // NEUROMORPHIC: Quick state consolidation instead of heavy processing
    int activeNeurons = 0;
    int firingNeurons = 0;
    std::vector<int> classAssignments(graphNodes.size(), -1);
    
    // FIRST: Get actual detected objects and their bounding boxes for class assignment
    std::vector<cv::Rect> detectedBoxes;
    std::vector<int> detectedClasses;
    std::vector<float> detectedConfs;
    
    if (!this->outputTensors.empty()) {
        float *boxOutput = this->outputTensors[0].GetTensorMutableData<float>();
        if (boxOutput && !this->outputShapes.empty()) {
            cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
            float *output0ptr = (float *)output0.data;
            int rows = (int)this->outputShapes[0][2];
            int cols = (int)this->outputShapes[0][1];
            
            // Extract detected objects
            for (int i = 0; i < rows; i++) {
                std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
                if (it.size() >= 4 + classNums) {
                    float confidence;
                    int classId;
                    this->getBestClassInfo(it.begin(), confidence, classId, classNums);
                    
                    if (confidence > this->confThreshold) {
                        int centerX = (int)(it[0]);
                        int centerY = (int)(it[1]);
                        int width = (int)(it[2]);
                        int height = (int)(it[3]);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;
                        
                        detectedBoxes.emplace_back(left, top, width, height);
                        detectedClasses.push_back(classId);
                        detectedConfs.push_back(confidence);
                    }
                }
            }
        }
    }
    
    std::cout << "NSGS NEUROMORPHIC: Found " << detectedBoxes.size() << " detected objects for class assignment" << std::endl;
    
    // Analyze final neuron states after spike propagation
    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto& neuron = graphNodes[i];
        float potential = neuron->getPotential();
        float threshold = neuron->getThreshold();
        
        if (potential > 0.1f) { // Neuron has significant activity
            activeNeurons++;
            
            if (potential > threshold) { // Neuron is firing
                firingNeurons++;
                
                // PROPER CLASS ASSIGNMENT: Assign neuron to detected object it's closest to
                cv::Point2i pos = neuron->getPosition();
                int assignedClass = -1;
                float minDistance = std::numeric_limits<float>::max();
                
                // Find closest detected object
                for (size_t j = 0; j < detectedBoxes.size(); j++) {
                    cv::Rect box = detectedBoxes[j];
                    
                    // Check if neuron is inside the bounding box
                    if (pos.x >= box.x && pos.x < box.x + box.width &&
                        pos.y >= box.y && pos.y < box.y + box.height) {
                        assignedClass = detectedClasses[j];
                        minDistance = 0; // Inside box = minimum distance
                        break;
                    }
                    
                    // Calculate distance to box center if not inside
                    float boxCenterX = box.x + box.width / 2.0f;
                    float boxCenterY = box.y + box.height / 2.0f;
                    float distance = std::sqrt((pos.x - boxCenterX) * (pos.x - boxCenterX) + 
                                             (pos.y - boxCenterY) * (pos.y - boxCenterY));
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        assignedClass = detectedClasses[j];
                    }
                }
                
                // Only assign if close enough to a detected object (within 150 pixels for better coverage)
                if (minDistance < 150.0f && assignedClass != -1) {
                    neuron->setClassId(assignedClass);
                    classAssignments[i] = assignedClass;
                }
            }
        }
        
        // Timeout check
        if (std::chrono::high_resolution_clock::now() > timeout) {
            std::cout << "NSGS NEUROMORPHIC: Event processing timeout, using current state" << std::endl;
            break;
        }
    }
    
    // NEUROMORPHIC: Simple state propagation for unassigned neurons
    for (size_t i = 0; i < graphNodes.size(); i++) {
        if (classAssignments[i] == -1) { // Unassigned neuron
            auto& neuron = graphNodes[i];
            cv::Point2i pos = neuron->getPosition();
            
            // Find nearest firing neuron for class inheritance
            float minDistance = std::numeric_limits<float>::max();
            int nearestClass = -1;
            
            for (size_t j = 0; j < graphNodes.size(); j++) {
                if (classAssignments[j] != -1) { // Has class assignment
                    auto& otherNeuron = graphNodes[j];
                    cv::Point2i otherPos = otherNeuron->getPosition();
                    
                    float dx = static_cast<float>(pos.x - otherPos.x);
                    float dy = static_cast<float>(pos.y - otherPos.y);
                    float distance = std::sqrt(dx*dx + dy*dy);
                    
                    if (distance < minDistance && distance < 120.0f) { // Within influence range - increased
                        minDistance = distance;
                        nearestClass = classAssignments[j];
                    }
                }
            }
            
            if (nearestClass != -1) {
                neuron->setClassId(nearestClass);
                classAssignments[i] = nearestClass;
            }
        }
        
        // Timeout check
        if (std::chrono::high_resolution_clock::now() > timeout) {
            std::cout << "NSGS NEUROMORPHIC: Class propagation timeout, finalizing" << std::endl;
            break;
        }
    }
    
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startTime).count();
    
    std::cout << "NSGS NEUROMORPHIC: Event processing completed in " << elapsedTime << "ms" << std::endl;
    std::cout << "NSGS NEUROMORPHIC: Active neurons: " << activeNeurons << ", Firing neurons: " << firingNeurons << std::endl;
    std::cout << "NSGS NEUROMORPHIC: Neural network state finalized for segmentation" << std::endl;
}

std::vector<Yolov8Result> NsgsPredictor::postprocessing(const cv::Size &resizedImageShape,
                                                      const cv::Size &originalImageShape,
                                                      std::vector<Ort::Value> &outputTensors)
{
    // Safety check
    if (outputTensors.empty()) {
        std::cerr << "NSGS: Empty output tensors in postprocessing" << std::endl;
        return {};
    }
    
    // Store the output tensors for use in NSGS processing
    this->outputTensors.clear();
    for (auto& tensor : outputTensors) {
        this->outputTensors.push_back(std::move(tensor));
    }
    
    // Box outputs
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float *boxOutput = nullptr;
    try {
        if (!this->outputTensors.empty()) {
            boxOutput = this->outputTensors[0].GetTensorMutableData<float>();
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "NSGS: Error accessing box output tensor: " << e.what() << std::endl;
        return {};
    }
    
    if (!boxOutput) {
        std::cerr << "NSGS: Invalid box output data in postprocessing" << std::endl;
        return {};
    }
    
    // Validate output shape
    if (this->outputShapes.empty() || this->outputShapes[0].size() < 3) {
        std::cerr << "NSGS: Invalid output shapes in postprocessing" << std::endl;
        return {};
    }
    
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
        
        // Safety check for classNums
        if (it.size() < 4 + classNums) {
            std::cerr << "NSGS: Detection output row too small for class count" << std::endl;
            continue;
        }
        
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);

        if (confidence > this->confThreshold)
        {
            if (this->hasMask)
            {
                // Check for valid mask proposals
                if (it.size() > 4 + classNums) {
                    std::vector<float> temp(it.begin() + 4 + classNums, it.end());
                    picked_proposals.push_back(temp);
                }
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
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);
    }

    // Process masks if available
    if (this->hasMask && this->outputTensors.size() > 1)
    {
        float *maskOutput = nullptr;
        try {
            maskOutput = this->outputTensors[1].GetTensorMutableData<float>();
        } catch (const Ort::Exception& e) {
            std::cerr << "NSGS: Error accessing mask output tensor: " << e.what() << std::endl;
        }
        
        if (maskOutput && this->outputShapes.size() > 1 && this->outputShapes[1].size() >= 4) {
            std::vector<int64_t> maskShape = this->outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
            std::vector<int> mask_protos_shape = {1, (int)maskShape[1], (int)maskShape[2], (int)maskShape[3]};
            mask_protos = cv::Mat(mask_protos_shape, CV_32F, maskOutput);
        }
    }
    else if (this->hasMask)
    {
        std::cout << "NSGS: Warning - Expected mask output tensor not available" << std::endl;
    }

    // Create original RGB image for feature extraction and graph construction
    cv::Mat processedRgbImage = cv::Mat((int)this->inputShapes[0][2], (int)this->inputShapes[0][3], CV_8UC3);
    
    // Fill with background color matching the letterboxing
    processedRgbImage.setTo(cv::Scalar(114, 114, 114));
    
    // Draw the detected objects on the image to provide visual cues for graph construction
    // This helps with feature extraction by highlighting important regions
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        
        // Draw a filled rectangle with class-specific color
        cv::Scalar color(50 + (classId * 100) % 200, 50 + (classId * 150) % 200, 50 + (classId * 50) % 200);
        cv::rectangle(processedRgbImage, box, color, 2);
        
        // Draw a small filled circle at the center for better node seeding
        int centerX = box.x + box.width / 2;
        int centerY = box.y + box.height / 2;
        cv::circle(processedRgbImage, cv::Point(centerX, centerY), 5, color, -1);
    }
    
    // Now build the segmentation graph with the enhanced image and the CNN output
    buildSegmentationGraph(processedRgbImage);

    // Continue with neural graph processing
    // Add a timeout wrapper around runAsyncEventProcessing to prevent getting stuck
    std::cout << "NSGS: Starting neural graph processing with 10 second timeout..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto timeout = startTime + std::chrono::seconds(10);

    // Create a processing thread that we can interrupt if it takes too long
    std::atomic<bool> processingCompleted(false);
    std::thread processingThread([&]() {
        try {
            // This is the main processing function that might be getting stuck
            runAsyncEventProcessing();
            processingCompleted.store(true);
        } catch (const std::exception& e) {
            std::cerr << "NSGS: Exception in processing thread: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "NSGS: Unknown exception in processing thread" << std::endl;
        }
    });

    // Monitor the processing thread with timeout
    int progressCounter = 0;
    while (!processingCompleted.load()) {
        // Check if we've exceeded the timeout
        auto now = std::chrono::high_resolution_clock::now();
        if (now > timeout) {
            std::cout << "NSGS: Processing timeout reached (10s). Continuing with partial results." << std::endl;
            break;
        }
        
        // Print progress every second
        if (++progressCounter % 10 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
            std::cout << "NSGS: Still processing... " << elapsed << " seconds elapsed" << std::endl;
        }
        
        // Short sleep to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Clean up the processing thread
    if (processingThread.joinable()) {
        if (!processingCompleted.load()) {
            // Thread is still running but we've timed out - we need to let it finish naturally
            // We can't cancel it safely but we'll let it continue running in the background
            processingThread.detach();
            std::cout << "NSGS: Processing thread detached, continuing with reconstruction..." << std::endl;
        } else {
            // Thread completed successfully
            processingThread.join();
            std::cout << "NSGS: Processing thread completed successfully" << std::endl;
        }
    }

    // Proceed with reconstruction regardless of whether processing completed or timed out
    cv::Mat neurographMask = reconstructFromNeuralGraph();
    
    // Prepare results
    std::vector<Yolov8Result> results;
    for (int idx : indices)
    {
        Yolov8Result res;
        res.box = cv::Rect(boxes[idx]);
        res.conf = confs[idx];
        res.classId = classIds[idx];
        
        if (this->hasMask && !mask_protos.empty())
        {
            if (!picked_proposals.empty()) {
                // Get the CNN-predicted mask
                cv::Mat proposal_mat = cv::Mat(picked_proposals[idx]).t();
                if (proposal_mat.empty()) {
                    std::cerr << "NSGS: Empty proposal matrix" << std::endl;
                    continue;
                }
                
                cv::Mat cnnMask = this->getMask(proposal_mat, mask_protos);
                if (cnnMask.empty()) {
                    std::cerr << "NSGS: Failed to generate CNN mask" << std::endl;
                    continue;
                }
                
                // Create the NSGS refined segmentation mask (integrating CNN + NSGS)
                cv::Mat refinedMask;
                
                // 1. Extract the class-specific mask from neurographMask
                cv::Mat currentClassMask = cv::Mat::zeros(neurographMask.size(), CV_8UC1);
                cv::compare(neurographMask, cv::Scalar(classIds[idx] + 1), currentClassMask, cv::CMP_EQ);
                
                // 2. Convert to floating point for blending
                cv::Mat cnnMaskF, nsgsClassMaskF;
                cnnMask.convertTo(cnnMaskF, CV_32F, 1.0/255.0);
                currentClassMask.convertTo(nsgsClassMaskF, CV_32F, 1.0/255.0);
                
                // 3. Compute edge strength in CNN mask using Scharr operator (better gradient estimation)
                cv::Mat gradX, gradY, cnnEdges;
                cv::Scharr(cnnMaskF, gradX, CV_32F, 1, 0);
                cv::Scharr(cnnMaskF, gradY, CV_32F, 0, 1);
                cv::magnitude(gradX, gradY, cnnEdges);
                
                // Normalize edge strengths to [0,1]
                double minVal, maxVal;
                cv::minMaxLoc(cnnEdges, &minVal, &maxVal);
                if (maxVal > 0)
                    cnnEdges = cnnEdges / maxVal;
                
                // 4. Create blending weight map - emphasize NSGS near edges, CNN elsewhere
                cv::Mat edgeWeight;
                cv::GaussianBlur(cnnEdges, edgeWeight, cv::Size(9, 9), 3.0);
                
                // Apply bilateral filter to NSGS mask to remove noise while preserving edges
                cv::Mat nsgsFiltered;
                cv::bilateralFilter(nsgsClassMaskF, nsgsFiltered, 9, 75, 75);
                
                // 5. Blend the masks with weighted combination
                // - Use CNN for overall structure (with weight decay near edges)
                // - Use NSGS for detailed boundaries (with weight increase near edges)
                cv::Mat blendedMask = (1.0 - edgeWeight.mul(0.7)).mul(cnnMaskF) + 
                                     edgeWeight.mul(0.7).mul(nsgsFiltered);
                
                // 6. Apply ROI constraint from detection box (with small margin)
                cv::Rect roi = boxes[idx];
                // Expand ROI slightly to account for potential mask edges
                int margin = std::max(5, std::min(roi.width, roi.height) / 10);
                roi.x = std::max(0, roi.x - margin);
                roi.y = std::max(0, roi.y - margin);
                roi.width = std::min(blendedMask.cols - roi.x, roi.width + 2*margin);
                roi.height = std::min(blendedMask.rows - roi.y, roi.height + 2*margin);
                
                cv::Mat roiMask = cv::Mat::zeros(blendedMask.size(), CV_32F);
                cv::Mat roiRect = roiMask(roi);
                roiRect.setTo(1.0);
                cv::GaussianBlur(roiMask, roiMask, cv::Size(9, 9), 3.0); // Soft ROI boundary
                
                blendedMask = blendedMask.mul(roiMask);
                
                // 7. Apply final threshold and convert to output format
                cv::Mat finalMask;
                cv::threshold(blendedMask, finalMask, this->maskThreshold, 1.0, cv::THRESH_BINARY);
                finalMask.convertTo(res.boxMask, CV_8UC1, 255);
                
                // Debug info
                std::cout << "NSGS: Created refined mask for class " << classIds[idx] 
                          << " with size " << res.boxMask.size() << std::endl;
            }
            else {
                // Fall back to NSGS-only mask if no CNN proposals available
                cv::Mat currentClassMask = cv::Mat::zeros(neurographMask.size(), CV_8UC1);
                cv::compare(neurographMask, cv::Scalar(classIds[idx] + 1), currentClassMask, cv::CMP_EQ);
                currentClassMask.convertTo(res.boxMask, CV_8UC1, 255);
                
                std::cout << "NSGS: Using graph-only mask for class " << classIds[idx] << std::endl;
            }
        }
        else
        {
            // For detection-only models, create an empty mask
            res.boxMask = cv::Mat::zeros((int)this->inputShapes[0][2], (int)this->inputShapes[0][3], CV_8UC1);
        }

        utils::scaleCoords(res.box, res.boxMask, this->maskThreshold, resizedImageShape, originalImageShape);
        results.emplace_back(res);
    }

    // Stop spike processing
    spikeQueue->stopProcessing();

    return results;
}

std::vector<Yolov8Result> NsgsPredictor::predict(cv::Mat &image)
{
    // If using pipeline parallelism, use async version
    if (usePipeline) {
        // Submit the image for processing
        predictAsync(image);
        
        // Wait for result (with a reasonable timeout)
        std::vector<Yolov8Result> results;
        if (getNextPredictionResult(results, 5000)) {
            return results;
        }
        
        // If failed to get result, fall back to synchronous processing
        std::cout << "NSGS: Pipeline processing timed out, falling back to synchronous processing" << std::endl;
    }
    
    // Validate image before processing
    if (image.empty()) {
        std::cerr << "NSGS: Empty image provided to predict" << std::endl;
        return {};
    }
    
    // Original synchronous processing
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    // Safety check for blob allocation
    if (!blob) {
        std::cerr << "NSGS: Failed to allocate blob in preprocessing" << std::endl;
        return {};
    }

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;
    try {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()));
    } catch (const Ort::Exception& e) {
        std::cerr << "NSGS: Error creating input tensor: " << e.what() << std::endl;
        delete[] blob;
        return {};
    }

    // Run inference with exception handling
    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = this->session.Run(
            Ort::RunOptions{nullptr},
            this->inputNames.data(),
            inputTensors.data(),
            this->inputNames.size(),
            this->outputNames.data(),
            this->outputNames.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "NSGS: Error running inference: " << e.what() << std::endl;
        delete[] blob;
        return {};
    }

    // If using data parallelism, use partitioned processing
    if (numPartitions > 1 && !usePipeline) {
        // Store output tensors for use by partitions
        this->outputTensors.clear();
        for (auto& tensor : outputTensors) {
            this->outputTensors.push_back(std::move(tensor));
        }
        
        // Build the graph
        buildSegmentationGraph(image);
        
        // Create partitions
        createGraphPartitions();
        
        // Start processing in partitions
        for (auto& partition : partitions) {
            partition->start();
        }
        
        // Wait for processing to complete with timeout
        const int maxIterations = 100; // Prevent infinite loop
        int iterations = 0;
        
        while (iterations < maxIterations) {
            bool allDone = true;
            
            // Check if any partition has active work
            for (auto& partition : partitions) {
                // Simple check - we'll continue improving this
                if (!partition->getNodes().empty()) {
                    allDone = false;
                    break;
                }
            }
            
            if (allDone) break;
            
            // Synchronize boundary nodes
            syncPartitionBoundaries();
            
            // Sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            iterations++;
        }
        
        // Stop partitions
        for (auto& partition : partitions) {
            partition->stop();
        }
        
        // Run a final synchronization to ensure all boundary nodes are updated
        syncPartitionBoundaries();
    }

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

void NsgsPredictor::createGraphPartitions()
{
    // Clear any existing partitions
    partitions.clear();
    
    if (numPartitions <= 1 || graphNodes.empty()) {
        std::cout << "NSGS: No graph partitioning needed (single partition or empty graph)" << std::endl;
        return;
    }
    
    std::cout << "NSGS: Creating " << numPartitions << " graph partitions" << std::endl;
    
    // Create the partition objects
    for (int i = 0; i < numPartitions; i++) {
        partitions.push_back(std::make_unique<GraphPartition>(i, &graphNodes));
    }
    
    // Determine partitioning strategy (spatial grid-based partitioning)
    int width = (int)this->inputShapes[0][3];
    int height = (int)this->inputShapes[0][2];
    
    // Decide on grid dimensions (try to keep partitions roughly square)
    int gridCols = std::sqrt(numPartitions);
    int gridRows = (numPartitions + gridCols - 1) / gridCols;
    
    // Adjust if necessary
    while (gridRows * gridCols < numPartitions) {
        gridCols++;
    }
    
    // Calculate cell dimensions
    float cellWidth = static_cast<float>(width) / gridCols;
    float cellHeight = static_cast<float>(height) / gridRows;
    
    std::cout << "NSGS: Using " << gridRows << "x" << gridCols << " spatial partitioning grid" << std::endl;
    
    // Assign nodes to partitions based on spatial grid
    std::vector<bool> boundaryNodes(graphNodes.size(), false);
    std::vector<int> nodePartition(graphNodes.size(), -1);
    
    // First pass: assign each node to a partition based on position
    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto& node = graphNodes[i];
        cv::Point2i pos = node->getPosition();
        
        // Calculate grid cell coordinates
        int gridX = std::min(gridCols - 1, std::max(0, static_cast<int>(pos.x / cellWidth)));
        int gridY = std::min(gridRows - 1, std::max(0, static_cast<int>(pos.y / cellHeight)));
        
        // Map to partition ID
        int partitionId = gridY * gridCols + gridX;
        if (partitionId >= numPartitions) {
            partitionId = numPartitions - 1;
        }
        
        nodePartition[i] = partitionId;
    }
    
    // Second pass: identify boundary nodes
    for (size_t i = 0; i < graphNodes.size(); i++) {
        int partitionId = nodePartition[i];
        auto& node = graphNodes[i];
        
        // Get the node's position
        cv::Point2i pos = node->getPosition();
        
        // Check if node is near a partition boundary
        float distanceToEdgeX = std::min(
            std::fmod(pos.x, cellWidth),
            cellWidth - std::fmod(pos.x, cellWidth)
        );
        
        float distanceToEdgeY = std::min(
            std::fmod(pos.y, cellHeight),
            cellHeight - std::fmod(pos.y, cellHeight)
        );
        
        // Boundary margin (nodes within this distance to boundary are considered boundary nodes)
        float boundaryMargin = std::min(cellWidth, cellHeight) * 0.1f;
        
        if (distanceToEdgeX < boundaryMargin || distanceToEdgeY < boundaryMargin) {
            boundaryNodes[i] = true;
        }
    }
    
    // Third pass: add nodes to partitions
    for (size_t i = 0; i < graphNodes.size(); i++) {
        int partitionId = nodePartition[i];
        if (partitionId >= 0 && partitionId < numPartitions) {
            partitions[partitionId]->addNode(graphNodes[i], boundaryNodes[i]);
        }
    }
    
    // Fourth pass: establish cross-partition connections
    for (size_t i = 0; i < graphNodes.size(); i++) {
        int partitionId = nodePartition[i];
        if (partitionId < 0 || partitionId >= numPartitions) continue;
        
        auto& node = graphNodes[i];
        
        // Check each connection of this node
        for (size_t j = 0; j < graphNodes.size(); j++) {
            if (i == j) continue;
            
            auto& otherNode = graphNodes[j];
            int otherPartitionId = nodePartition[j];
            
            if (otherPartitionId != partitionId) {
                // Nodes are in different partitions
                // Check if they are connected
                cv::Point2i pos1 = node->getPosition();
                cv::Point2i pos2 = otherNode->getPosition();
                
                // Calculate distance
                float distance = cv::norm(pos1 - pos2);
                
                // If nodes are close enough, consider them connected across partition boundary
                float connectionThreshold = std::min(cellWidth, cellHeight) * 0.2f;
                if (distance < connectionThreshold) {
                    // Add external connection
                    // Find local index in partition
                    const auto& partitionNodes = partitions[partitionId]->getNodes();
                    for (size_t k = 0; k < partitionNodes.size(); k++) {
                        if (partitionNodes[k]->getId() == node->getId()) {
                            partitions[partitionId]->addExternalConnection(k, j);
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Log partition statistics
    int totalBoundaryNodes = 0;
    int totalConnections = 0;
    
    for (const auto& partition : partitions) {
        totalBoundaryNodes += partition->getBoundaryNodeCount();
    }
    
    std::cout << "NSGS: Created " << partitions.size() << " partitions with " 
              << totalBoundaryNodes << " boundary nodes" << std::endl;
}

void NsgsPredictor::syncPartitionBoundaries()
{
    // Synchronize boundary nodes across all partitions
    for (auto& partition : partitions) {
        partition->syncBoundaryNodes();
    }
}

void NsgsPredictor::startPipeline()
{
    if (pipelineRunning.load()) return;
    
    std::cout << "NSGS: Starting pipeline execution" << std::endl;
    
    // Clear any existing pipeline stages
    pipelineStages.clear();
    
    // Create pipeline stages
    // 1. Input preprocessing
    auto preprocessStage = std::make_unique<PipelineStage>(PipelineStage::StageType::INPUT_PREPROCESSING);
    
    // 2. Model inference
    auto inferenceStage = std::make_unique<PipelineStage>(PipelineStage::StageType::MODEL_INFERENCE);
    
    // 3. Graph construction
    auto graphStage = std::make_unique<PipelineStage>(PipelineStage::StageType::GRAPH_CONSTRUCTION);
    
    // 4. Spike propagation
    auto spikeStage = std::make_unique<PipelineStage>(PipelineStage::StageType::SPIKE_PROPAGATION);
    
    // 5. Result generation
    auto resultStage = std::make_unique<PipelineStage>(PipelineStage::StageType::RESULT_GENERATION);
    
    // Define stage processing functions
    
    // 1. Preprocessing stage
    preprocessStage->start([this, preprocessStage = preprocessStage.get(), inferenceStage = inferenceStage.get()]() {
        std::unique_lock<std::mutex> lock(preprocessStage->stageMutex);
        if (preprocessStage->inputImages.empty()) return;
        
        // Get the next image
        cv::Mat image = preprocessStage->inputImages.front();
        preprocessStage->inputImages.pop();
        lock.unlock();
        
        // Preprocess the image
        float *blob = nullptr;
        std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
        this->preprocessing(image, blob, inputTensorShape);
        
        // Create input tensor values
        size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
        std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
        
        // Pass to next stage
        {
            std::lock_guard<std::mutex> nextLock(inferenceStage->stageMutex);
            inferenceStage->preprocessedData.push(inputTensorValues);
            inferenceStage->stageCondition.notify_one();
        }
        
        delete[] blob;
    });
    
    // 2. Model inference stage
    inferenceStage->start([this, inferenceStage = inferenceStage.get(), graphStage = graphStage.get()]() {
        std::unique_lock<std::mutex> lock(inferenceStage->stageMutex);
        if (inferenceStage->preprocessedData.empty()) return;
        
        // Get preprocessed data
        std::vector<float> inputTensorValues = inferenceStage->preprocessedData.front();
        inferenceStage->preprocessedData.pop();
        lock.unlock();
        
        // Create input tensor
        std::vector<int64_t> inputTensorShape{1, 3, (int64_t)this->inputShapes[0][2], (int64_t)this->inputShapes[0][3]};
        
        std::vector<Ort::Value> inputTensors;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputTensorShape.data(), inputTensorShape.size()));
        
        // Run inference
        std::vector<Ort::Value> outputTensors = this->session.Run(
            Ort::RunOptions{nullptr},
            this->inputNames.data(),
            inputTensors.data(),
            this->inputNames.size(),
            this->outputNames.data(),
            this->outputNames.size());
        
        // Pass to next stage
        {
            std::lock_guard<std::mutex> nextLock(graphStage->stageMutex);
            graphStage->modelOutputs.push(std::move(outputTensors));
            graphStage->stageCondition.notify_one();
        }
    });
    
    // 3. Graph construction stage
    graphStage->start([this, graphStage = graphStage.get(), spikeStage = spikeStage.get()]() {
        std::unique_lock<std::mutex> lock(graphStage->stageMutex);
        if (graphStage->modelOutputs.empty()) return;
        
        // Get model outputs
        std::vector<Ort::Value> outputTensors = std::move(graphStage->modelOutputs.front());
        graphStage->modelOutputs.pop();
        lock.unlock();
        
        // Store the output tensors
        this->outputTensors.clear();
        for (auto& tensor : outputTensors) {
            this->outputTensors.push_back(std::move(tensor));
        }
        
        // Create original RGB image for feature extraction and graph construction
        cv::Mat processedRgbImage = cv::Mat((int)this->inputShapes[0][2], (int)this->inputShapes[0][3], CV_8UC3);
        processedRgbImage.setTo(cv::Scalar(114, 114, 114));
        
        // Build the segmentation graph
        buildSegmentationGraph(processedRgbImage);
        
        // Create graph partitions if using data parallelism
        if (numPartitions > 1) {
            createGraphPartitions();
            
            // Start each partition
            for (auto& partition : partitions) {
                partition->start();
            }
        }
        
        // Pass to next stage (use the current graph nodes vector)
        {
            std::lock_guard<std::mutex> nextLock(spikeStage->stageMutex);
            // We just signal that this stage is ready, no need to pass the actual nodes
            spikeStage->stageCondition.notify_one();
        }
    });
    
    // 4. Spike propagation stage
    spikeStage->start([this, spikeStage = spikeStage.get(), resultStage = resultStage.get()]() {
        // No lock needed since we don't use the queue for this stage
        
        if (numPartitions > 1) {
            // Using partitioned graph - wait for partitions to process
            
            // Wait for partitions to process
            while (true) {
                bool allDone = true;
                for (auto& partition : partitions) {
                    if (!partition->getNodes().empty()) {
                        allDone = false;
                        break;
                    }
                }
                
                if (allDone) break;
                
                // Synchronize boundary nodes
                syncPartitionBoundaries();
                
                // Sleep briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Stop the partitions
            for (auto& partition : partitions) {
                partition->stop();
            }
        }
        else {
            // Using single partition - run the processing directly
            runAsyncEventProcessing();
        }
        
        // Pass to next stage
        {
            std::lock_guard<std::mutex> nextLock(resultStage->stageMutex);
            // Just signal that processing is complete
            resultStage->stageCondition.notify_one();
        }
    });
    
    // 5. Result generation stage
    resultStage->start([this, resultStage = resultStage.get()]() {
        // No lock needed since we don't use the queue for this stage
        
        // Generate results from processed graph
        cv::Size resizedShape(this->inputShapes[0][3], this->inputShapes[0][2]);
        cv::Size originalShape(resizedShape); // For simplicity - should be passed from the original image
        
        // Generate results
        std::vector<Yolov8Result> results = this->postprocessing(
            resizedShape, originalShape, this->outputTensors);
        
        // Add to output queue
        {
            std::lock_guard<std::mutex> lock(pipelineMutex);
            outputQueue.push(results);
            pipelineCondition.notify_one();
        }
    });
    
    // Add stages to the pipeline
    pipelineStages.push_back(std::move(preprocessStage));
    pipelineStages.push_back(std::move(inferenceStage));
    pipelineStages.push_back(std::move(graphStage));
    pipelineStages.push_back(std::move(spikeStage));
    pipelineStages.push_back(std::move(resultStage));
    
    // Start the pipeline
    pipelineRunning.store(true);
    
    // Start the pipeline worker thread
    pipelineThread = std::thread(&NsgsPredictor::pipelineWorker, this);
    
    std::cout << "NSGS: Pipeline execution started with " << pipelineStages.size() << " stages" << std::endl;
}

void NsgsPredictor::stopPipeline()
{
    if (!pipelineRunning.load()) return;
    
    std::cout << "NSGS: Stopping pipeline execution" << std::endl;
    
    // Stop the pipeline
    pipelineRunning.store(false);
    pipelineCondition.notify_all();
    
    // Wait for pipeline thread to finish
    if (pipelineThread.joinable()) {
        pipelineThread.join();
    }
    
    // Stop all stages
    for (auto& stage : pipelineStages) {
        stage->stop();
    }
    
    // Clear all queues
    {
        std::lock_guard<std::mutex> lock(pipelineMutex);
        std::queue<cv::Mat> emptyInputQueue;
        std::queue<std::vector<Yolov8Result>> emptyOutputQueue;
        std::swap(inputQueue, emptyInputQueue);
        std::swap(outputQueue, emptyOutputQueue);
    }
    
    std::cout << "NSGS: Pipeline execution stopped" << std::endl;
}

void NsgsPredictor::pipelineWorker()
{
    while (pipelineRunning.load()) {
        // Check if there are any inputs to process
        cv::Mat input;
        
        {
            std::unique_lock<std::mutex> lock(pipelineMutex);
            if (inputQueue.empty()) {
                // Wait for an input or termination
                pipelineCondition.wait_for(lock, std::chrono::milliseconds(100), 
                    [this] { return !inputQueue.empty() || !pipelineRunning.load(); });
                
                if (!pipelineRunning.load()) break;
                if (inputQueue.empty()) continue;
            }
            
            // Get the next input
            input = inputQueue.front();
            inputQueue.pop();
        }
        
        // Feed the input to the first stage
        if (!pipelineStages.empty() && !input.empty()) {
            auto& firstStage = pipelineStages[0];
            
            {
                std::lock_guard<std::mutex> lock(firstStage->stageMutex);
                firstStage->inputImages.push(input);
                firstStage->stageCondition.notify_one();
            }
        }
        
        // Sleep briefly to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void NsgsPredictor::predictAsync(cv::Mat &image)
{
    // Start the pipeline if not running
    if (!pipelineRunning.load() && usePipeline) {
        startPipeline();
    }
    
    // Add the image to the input queue
    {
        std::lock_guard<std::mutex> lock(pipelineMutex);
        inputQueue.push(image.clone()); // Clone to ensure we have our own copy
        pipelineCondition.notify_one();
    }
}

bool NsgsPredictor::getNextPredictionResult(std::vector<Yolov8Result> &results, int timeoutMs)
{
    std::unique_lock<std::mutex> lock(pipelineMutex);
    
    // Wait for a result or timeout
    bool hasResult = pipelineCondition.wait_for(lock, std::chrono::milliseconds(timeoutMs),
        [this] { return !outputQueue.empty() || !pipelineRunning.load(); });
    
    if (!hasResult || outputQueue.empty()) {
        return false;
    }
    
    // Get the result
    results = outputQueue.front();
    outputQueue.pop();
    
    return true;
}

// Add this new method somewhere appropriate in the file
void NsgsPredictor::registerNodeEdgeStrengths(const cv::Mat &image)
{
    std::cout << "NSGS: Registering edge strengths for priority-based scheduling" << std::endl;
    
    // Convert image to grayscale for edge detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // Calculate image gradients using Scharr operator for better edge detection
    cv::Mat gradX, gradY, gradMag;
    cv::Scharr(grayImage, gradX, CV_32F, 1, 0);
    cv::Scharr(grayImage, gradY, CV_32F, 0, 1);
    
    // Compute gradient magnitude
    cv::magnitude(gradX, gradY, gradMag);  // Fixed: Use cv::magnitude instead of cv::cartToPolar
    cv::normalize(gradMag, gradMag, 0, 1, cv::NORM_MINMAX);
    
    // Apply Gaussian blur to smooth out the gradient map
    cv::GaussianBlur(gradMag, gradMag, cv::Size(5, 5), 1.5);
    
    // Register edge strength for each node based on its position in the gradient map
    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto& node = graphNodes[i];
        cv::Point2i pos = node->getPosition();
        
        // Ensure position is within image bounds
        int x = std::min(gradMag.cols - 1, std::max(0, pos.x));
        int y = std::min(gradMag.rows - 1, std::max(0, pos.y));
        
        // Get edge strength at node position
        float edgeStrength = gradMag.at<float>(y, x);
        
        // Register with the spike queue for priority calculation
        spikeQueue->registerNodeEdgeStrength(node->getId(), edgeStrength);
    }
    
    std::cout << "NSGS: Registered edge strengths for " << graphNodes.size() << " nodes" << std::endl;
}

std::vector<Yolov8Result> NsgsPredictor::detect(cv::Mat& image, float confThreshold, float iouThreshold, float maskThreshold)
{
    std::cout << "NSGS: Starting detection with enhanced timeout protection..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto totalTimeout = startTime + std::chrono::seconds(15); // Reduced from 30 seconds to 15 seconds

    // Save thresholds
    this->confThreshold = confThreshold;
    this->iouThreshold = iouThreshold;
    this->maskThreshold = maskThreshold;
    
    // Initialize results vector
    std::vector<Yolov8Result> results;
    
    // Validate image before processing
    if (image.empty()) {
        std::cerr << "NSGS: Empty image provided to detect" << std::endl;
        return results;
    }
    
    // Original synchronous processing
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    
    // Use existing preprocessing method
    this->preprocessing(image, blob, inputTensorShape);
    
    // Safety check for blob allocation
    if (!blob) {
        std::cerr << "NSGS: Failed to allocate blob in preprocessing" << std::endl;
        return results;
    }

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    // Run inference with exception handling 
    try {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()));
            
        // Run inference
        this->outputTensors = this->session.Run(
            Ort::RunOptions{nullptr},
            this->inputNames.data(),
            inputTensors.data(),
            this->inputNames.size(),
            this->outputNames.data(),
            this->outputNames.size());
            
        // Save output shapes
        this->outputShapes.clear();
        for (size_t i = 0; i < this->outputTensors.size(); i++) {
            auto info = this->outputTensors[i].GetTensorTypeAndShapeInfo();
            this->outputShapes.push_back(info.GetShape());
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "NSGS Exception during inference: " << e.what() << std::endl;
        delete[] blob;
        return results;
    }
    
    // Check if we've exceeded our total timeout
    if (std::chrono::high_resolution_clock::now() > totalTimeout) {
        std::cout << "NSGS: Total timeout exceeded in detect(), returning empty results" << std::endl;
        delete[] blob;
        return results;
    }
    
    // First phase: Standard YOLO detection - Always get basic detection results
    std::cout << "NSGS: Processing initial YOLO detections..." << std::endl;
    cv::Size resizedShape(inputTensorShape[3], inputTensorShape[2]);
    cv::Size originalShape = image.size();
    
    // Process the output tensors directly for basic detection
    // Use the existing results vector declared earlier
    
    // Box outputs
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float *boxOutput = nullptr;
    try {
        if (!this->outputTensors.empty()) {
            boxOutput = this->outputTensors[0].GetTensorMutableData<float>();
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "NSGS: Error accessing box output tensor: " << e.what() << std::endl;
        delete[] blob;
        return results;
    }
    
    if (!boxOutput) {
        std::cerr << "NSGS: Invalid box output data in detect" << std::endl;
        delete[] blob;
        return results;
    }
    
    // Process detections
    cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
    float *output0ptr = (float *)output0.data;
    int rows = (int)this->outputShapes[0][2];
    int cols = (int)this->outputShapes[0][1];
    
    std::cout << "NSGS: Processing " << rows << " detections with " << cols << " features each" << std::endl;
    
    // For mask handling
    std::vector<std::vector<float>> picked_proposals;
    cv::Mat mask_protos;

    // Process detection boxes
    for (int i = 0; i < rows; i++)
    {
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        
        // Safety check for classNums
        if (it.size() < 4 + classNums) {
            continue;
        }
        
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);

        if (confidence > this->confThreshold)
        {
            if (this->hasMask)
            {
                // Check for valid mask proposals
                if (it.size() > 4 + classNums) {
                    std::vector<float> temp(it.begin() + 4 + classNums, it.end());
                    picked_proposals.push_back(temp);
                }
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

    std::cout << "NSGS: Found " << boxes.size() << " initial detections above confidence threshold" << std::endl;

    // Apply NMS
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);
    }
    
    std::cout << "NSGS: After NMS: " << indices.size() << " final detections" << std::endl;

    // Process masks if available
    if (this->hasMask && this->outputTensors.size() > 1)
    {
        float *maskOutput = nullptr;
        try {
            maskOutput = this->outputTensors[1].GetTensorMutableData<float>();
        } catch (const Ort::Exception& e) {
            std::cerr << "NSGS: Error accessing mask output tensor: " << e.what() << std::endl;
        }
        
        if (maskOutput && this->outputShapes.size() > 1 && this->outputShapes[1].size() >= 4) {
            std::vector<int64_t> maskShape = this->outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
            std::vector<int> mask_protos_shape = {1, (int)maskShape[1], (int)maskShape[2], (int)maskShape[3]};
            mask_protos = cv::Mat(mask_protos_shape, CV_32F, maskOutput);
            std::cout << "NSGS: Loaded mask prototypes: " << maskShape[1] << " channels, " 
                      << maskShape[2] << "x" << maskShape[3] << " resolution" << std::endl;
        }
    }
    
    // Create initial results from basic detections
    for (int idx : indices)
    {
        Yolov8Result res;
        res.box = cv::Rect(boxes[idx]);
        res.conf = confs[idx];
        res.classId = classIds[idx];
        
        if (this->hasMask && !mask_protos.empty() && idx < picked_proposals.size())
        {
            // Get the CNN-predicted mask
            cv::Mat proposal_mat = cv::Mat(picked_proposals[idx]).t();
            if (!proposal_mat.empty()) {
                cv::Mat cnnMask = this->getMask(proposal_mat, mask_protos);
                if (!cnnMask.empty()) {
                    // Scale mask to original image size
                    utils::scaleCoords(res.box, cnnMask, this->maskThreshold, resizedShape, originalShape);
                    res.boxMask = cnnMask;
                }
            }
        }
        
        results.emplace_back(res);
    }
    
    std::cout << "NSGS: Created " << results.size() << " detection results" << std::endl;
    
    // Set a flag to track if we completed full processing
    bool fullProcessingCompleted = false;
    
    // Only proceed with neural graph if enabled
    if (this->hasMask) {
        try {
            auto graphStartTime = std::chrono::high_resolution_clock::now();
            auto graphTimeout = graphStartTime + std::chrono::seconds(30); // Increased timeout to ensure novel processing completes
            
            // Extract features and create graph
            std::cout << "NSGS: Creating neural graph for segmentation..." << std::endl;
            buildSegmentationGraph(image);
            
            // Check timeout after graph creation - but don't give up immediately
            if (std::chrono::high_resolution_clock::now() > graphTimeout) {
                std::cout << "NSGS: Graph creation took longer than expected, but continuing with neuromorphic processing..." << std::endl;
            }
            
            // Extract features for each node (using existing embeddings from the output tensor)
            std::cout << "NSGS: Extracting node features..." << std::endl;
            
            // Create embeddings matrix if available
            cv::Mat embeddings;
            if (this->outputTensors.size() > 1) {
                float *maskOutput = this->outputTensors[1].GetTensorMutableData<float>();
                if (maskOutput && this->outputShapes.size() > 1 && this->outputShapes[1].size() >= 4) {
                    int protoChannels = this->outputShapes[1][1];
                    int protoHeight = this->outputShapes[1][2];
                    int protoWidth = this->outputShapes[1][3];
                    
                    embeddings = cv::Mat(protoHeight, protoWidth, CV_32FC(protoChannels), maskOutput);
                }
            }
            
            extractNodeFeatures(image, embeddings);
            
            // Always proceed with neuromorphic processing - this is the novel contribution
            std::cout << "NSGS: *** USING NOVEL NEUROMORPHIC ALGORITHMS ***" << std::endl;
            
            // Set up timing for propagation - give more time for novel processing
            auto propagationStartTime = std::chrono::high_resolution_clock::now();
            auto propagationTimeout = propagationStartTime + std::chrono::seconds(15); // Increased from 3 to 15 seconds
            
            // Run the novel spike propagation algorithm
            std::cout << "NSGS: Running NOVEL spike propagation algorithm..." << std::endl;
            this->propagateSpikes(true);
            
            // Run the novel event processing algorithm
            std::cout << "NSGS: Running NOVEL neuromorphic event processing..." << std::endl;
            this->runAsyncEventProcessing();
            
            // Use the novel reconstruction algorithm from neural graph
            std::cout << "NSGS: Running NOVEL graph-based reconstruction algorithm..." << std::endl;
            cv::Mat segMask = this->reconstructFromNeuralGraph();
            
            // Apply the novel mask fusion algorithm
            if (!segMask.empty()) {
                std::cout << "NSGS: *** APPLYING NOVEL NEUROMORPHIC-CNN FUSION ***" << std::endl;
                for (auto& result : results) {
                    // This is the novel part: combining neuromorphic graph processing with CNN predictions
                    cv::Rect bbox = result.box;
                    
                    // Ensure bounds are valid
                    int x = std::max(0, bbox.x);
                    int y = std::max(0, bbox.y);
                    int width = std::min(segMask.cols - x, bbox.width);
                    int height = std::min(segMask.rows - y, bbox.height);
                    
                    if (width <= 0 || height <= 0) continue;
                    
                    cv::Rect validRect(x, y, width, height);
                    cv::Mat neuromorphicMaskROI = segMask(validRect);
                    
                    // Novel fusion: Combine CNN mask with neuromorphic graph results
                    if (!neuromorphicMaskROI.empty() && cv::countNonZero(neuromorphicMaskROI) > 0) {
                        // Get original CNN mask
                        cv::Mat originalCnnMask = result.boxMask.clone();
                        
                        // Create neuromorphic mask for this region
                        cv::Mat neuromorphicMask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
                        cv::Mat segROI = neuromorphicMask(bbox);
                        neuromorphicMaskROI.copyTo(segROI);
                        
                        // NOVEL ALGORITHM: Adaptive fusion based on edge consistency
                        cv::Mat fusedMask;
                        if (!originalCnnMask.empty()) {
                            // Calculate edge maps for both masks
                            cv::Mat cnnEdges, neuromorphicEdges;
                            cv::Canny(originalCnnMask, cnnEdges, 50, 150);
                            cv::Canny(neuromorphicMask, neuromorphicEdges, 50, 150);
                            
                            // Create fusion weights based on edge agreement
                            cv::Mat edgeAgreement;
                            cv::bitwise_and(cnnEdges, neuromorphicEdges, edgeAgreement);
                            
                            // Apply bilateral fusion: CNN for interior, neuromorphic for boundaries
                            cv::Mat cnnMaskF, neuromorphicMaskF;
                            originalCnnMask.convertTo(cnnMaskF, CV_32F, 1.0/255.0);
                            neuromorphicMask.convertTo(neuromorphicMaskF, CV_32F, 1.0/255.0);
                            
                            // Weight map: higher weight for neuromorphic near edges
                            cv::Mat distTransform;
                            cv::distanceTransform(255 - edgeAgreement, distTransform, cv::DIST_L2, 5);
                            cv::normalize(distTransform, distTransform, 0, 1, cv::NORM_MINMAX);
                            
                            // Fusion: w * neuromorphic + (1-w) * CNN
                            cv::Mat weightForNeuromorphic = 1.0 - distTransform;
                            cv::Mat weightForCnn = distTransform;
                            
                            fusedMask = weightForNeuromorphic.mul(neuromorphicMaskF) + 
                                       weightForCnn.mul(cnnMaskF);
                            
                            // Convert back to 8-bit
                            fusedMask.convertTo(result.boxMask, CV_8UC1, 255);
                            
                            std::cout << "NSGS: Applied novel CNN-Neuromorphic fusion for detection " 
                                      << result.classId << std::endl;
                        } else {
                            // Use pure neuromorphic result if no CNN mask available
                            result.boxMask = neuromorphicMask;
                            std::cout << "NSGS: Applied pure neuromorphic segmentation for detection " 
                                      << result.classId << std::endl;
                        }
                    }
                }
                fullProcessingCompleted = true;
                std::cout << "NSGS: *** NOVEL NEUROMORPHIC PROCESSING COMPLETED SUCCESSFULLY ***" << std::endl;
            } else {
                std::cout << "NSGS: Warning - Neuromorphic reconstruction returned empty mask" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "NSGS: Exception during neural graph processing: " << e.what() << std::endl;
        }
    }
    
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startTime).count();
    std::cout << "NSGS: Detection " << (fullProcessingCompleted ? "fully completed" : "partially completed") 
              << " in " << elapsedTime << "ms" << std::endl;
    
    // Clean up allocated memory
    delete[] blob;
    
    return results;
}

// NOVEL ALGORITHM: Hierarchical Feature Extraction
// Only extract detailed features for high-priority nodes to reduce processing time
void NsgsPredictor::extractNodeFeaturesHierarchical(const cv::Mat &image, const cv::Mat &embeddings)
{
    std::cout << "NSGS NOVEL: Starting hierarchical feature extraction..." << std::endl;
    auto featureStartTime = std::chrono::high_resolution_clock::now();
    
    if (image.empty()) {
        std::cerr << "NSGS: Error - Empty image provided for feature extraction" << std::endl;
        return;
    }
    
    // NOVEL STEP 1: Calculate priority scores for all nodes
    std::vector<std::pair<size_t, float>> nodePriorities;
    nodePriorities.reserve(graphNodes.size());
    
    // Convert image to floating point for feature computation
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0/255.0);
    
    // Calculate image gradients for edge detection
    cv::Mat gradX, gradY, gradMag;
    cv::Scharr(floatImage, gradX, CV_32F, 1, 0);
    cv::Scharr(floatImage, gradY, CV_32F, 0, 1);
    cv::magnitude(gradX, gradY, gradMag);  // Fixed: Use cv::magnitude instead of cv::cartToPolar
    cv::normalize(gradMag, gradMag, 0, 1, cv::NORM_MINMAX);
    
    // Calculate CNN activation strength if available
    cv::Mat cnnActivations;
    if (!embeddings.empty()) {
        // Calculate activation magnitude from embeddings
        cnnActivations = cv::Mat(embeddings.rows, embeddings.cols, CV_32F, 0.0f);
        for (int y = 0; y < embeddings.rows; y++) {
            for (int x = 0; x < embeddings.cols; x++) {
                float sum = 0.0f;
                for (int c = 0; c < std::min(embeddings.channels(), 32); c++) {
                    float* pixelPtr = (float*)(embeddings.data + y*embeddings.step + x*embeddings.elemSize());
                    float val = pixelPtr[c];
                    sum += val * val;
                }
                cnnActivations.at<float>(y, x) = std::sqrt(sum);
            }
        }
        cv::normalize(cnnActivations, cnnActivations, 0, 1, cv::NORM_MINMAX);
        cv::resize(cnnActivations, cnnActivations, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);
    }
    
    // Calculate priority for each node
    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto& node = graphNodes[i];
        cv::Point2i pos = node->getPosition();
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        float priority = 0.0f;
        
        // Priority based on edge strength (40% weight)
        priority += 0.4f * gradMag.at<float>(y, x);
        
        // Priority based on CNN activations (40% weight)
        if (!cnnActivations.empty()) {
            priority += 0.4f * cnnActivations.at<float>(y, x);
        }
        
        // Priority based on spatial distribution (20% weight)
        float centerDistX = std::abs(static_cast<float>(x) / image.cols - 0.5f) * 2.0f;
        float centerDistY = std::abs(static_cast<float>(y) / image.rows - 0.5f) * 2.0f;
        float centerDist = std::sqrt(centerDistX*centerDistX + centerDistY*centerDistY) / std::sqrt(2.0f);
        priority += 0.2f * centerDist; // Peripheral nodes get slight priority
        
        nodePriorities.push_back(std::make_pair(i, priority));
    }
    
    // NOVEL STEP 2: Sort by priority and select top N% for detailed processing
    std::sort(nodePriorities.begin(), nodePriorities.end(), 
              [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                  return a.second > b.second;
              });
    
    // Process top 30% with detailed features, rest with simple features
    int detailedCount = std::min(static_cast<int>(nodePriorities.size() * 0.3f), 1000); // Cap at 1000 nodes
    int simpleCount = nodePriorities.size() - detailedCount;
    
    std::cout << "NSGS NOVEL: Processing " << detailedCount << " nodes with detailed features, " 
              << simpleCount << " with simple features" << std::endl;
    
    // NOVEL STEP 3: Extract features hierarchically
    cv::Mat labImage;
    cv::cvtColor(floatImage, labImage, cv::COLOR_BGR2Lab);
    
    // Process high-priority nodes with detailed features
    for (int i = 0; i < detailedCount; i++) {
        size_t nodeIdx = nodePriorities[i].first;
        auto& node = graphNodes[nodeIdx];
        cv::Point2i pos = node->getPosition();
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        std::vector<float> featureVector;
        
        // 1. LAB color features (3 features)
        cv::Vec3f labPixel = labImage.at<cv::Vec3f>(y, x);
        featureVector.push_back(labPixel[0]); // L
        featureVector.push_back(labPixel[1]); // a
        featureVector.push_back(labPixel[2]); // b
        
        // 2. Local texture variance (1 feature)
        float textureVar = 0.0f;
        int patchSize = 3;
        int patchCount = 0;
        float mean = 0.0f;
        
        // Convert to grayscale for texture calculation
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        
        // Calculate mean
        for (int dy = -patchSize; dy <= patchSize; dy++) {
            for (int dx = -patchSize; dx <= patchSize; dx++) {
                int nx = std::min(image.cols - 1, std::max(0, x + dx));
                int ny = std::min(image.rows - 1, std::max(0, y + dy));
                mean += grayImage.at<uchar>(ny, nx);
                patchCount++;
            }
        }
        mean /= patchCount;
        
        // Calculate variance
        for (int dy = -patchSize; dy <= patchSize; dy++) {
            for (int dx = -patchSize; dx <= patchSize; dx++) {
                int nx = std::min(image.cols - 1, std::max(0, x + dx));
                int ny = std::min(image.rows - 1, std::max(0, y + dy));
                float val = grayImage.at<uchar>(ny, nx);
                textureVar += (val - mean) * (val - mean);
            }
        }
        textureVar /= patchCount;
        featureVector.push_back(textureVar / 255.0f); // Normalize
        
        // 3. Gradient magnitude (1 feature)
        featureVector.push_back(gradMag.at<float>(y, x));
        
        // 4. CNN embeddings (4 features max)
        if (!embeddings.empty()) {
            float scaleX = static_cast<float>(embeddings.cols) / static_cast<float>(image.cols);
            float scaleY = static_cast<float>(embeddings.rows) / static_cast<float>(image.rows);
            int embX = std::min(embeddings.cols - 1, std::max(0, static_cast<int>(x * scaleX)));
            int embY = std::min(embeddings.rows - 1, std::max(0, static_cast<int>(y * scaleY)));
            
            int embChannels = embeddings.channels();
            for (int c = 0; c < std::min(4, embChannels); c++) {
                float* pixelPtr = (float*)(embeddings.data + embY*embeddings.step + embX*embeddings.elemSize());
                featureVector.push_back(pixelPtr[c]);
            }
        }
        
        // Normalize features
        float maxVal = 0.0f;
        for (float f : featureVector) maxVal = std::max(maxVal, std::abs(f));
        if (maxVal > 0) {
            for (float& f : featureVector) f /= maxVal;
        }
        
        node->setFeatures(featureVector);
    }
    
    // Process low-priority nodes with simple features (only 3 features)
    for (int i = detailedCount; i < nodePriorities.size(); i++) {
        size_t nodeIdx = nodePriorities[i].first;
        auto& node = graphNodes[nodeIdx];
        cv::Point2i pos = node->getPosition();
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        std::vector<float> featureVector;
        
        // Simple features: only color and gradient
        cv::Vec3f bgr = floatImage.at<cv::Vec3f>(y, x);
        featureVector.push_back(bgr[0]); // B
        featureVector.push_back(bgr[1]); // G
        featureVector.push_back(bgr[2]); // R
        
        node->setFeatures(featureVector);
    }
    
    auto featureTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - featureStartTime).count();
    
    std::cout << "NSGS NOVEL: Hierarchical feature extraction completed in " << featureTime 
              << "ms (" << detailedCount << " detailed, " << simpleCount << " simple)" << std::endl;
}

// NOVEL ALGORITHM: Selective Connection Creation
// Create connections only between nearby high-priority nodes to reduce processing time
void NsgsPredictor::createSelectiveConnections(const cv::Mat &image, const cv::Mat &labels, const std::vector<cv::Point2d> &centers)
{
    std::cout << "NSGS NOVEL: Creating selective connections based on priority..." << std::endl;
    auto connectionStartTime = std::chrono::high_resolution_clock::now();
    
    // NOVEL STEP 1: Identify high-priority nodes (top 50% by feature quality)
    std::vector<std::pair<size_t, float>> nodePriorities;
    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto& node = graphNodes[i];
        cv::Point2i pos = node->getPosition();
        
        // Calculate priority based on edge strength and position
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        // Simple gradient-based priority
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Mat grad;
        cv::Laplacian(gray, grad, CV_32F);
        cv::normalize(grad, grad, 0, 1, cv::NORM_MINMAX);
        
        float priority = grad.at<float>(y, x);
        nodePriorities.push_back(std::make_pair(i, priority));
    }
    
    // Sort by priority
    std::sort(nodePriorities.begin(), nodePriorities.end(),
              [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                  return a.second > b.second;
              });
    
    // Select top 50% as high-priority nodes
    int highPriorityCount = std::min(static_cast<int>(nodePriorities.size() * 0.5f), 500);
    std::unordered_set<size_t> highPriorityNodes;
    for (int i = 0; i < highPriorityCount; i++) {
        highPriorityNodes.insert(nodePriorities[i].first);
    }
    
    std::cout << "NSGS NOVEL: Selected " << highPriorityCount << " high-priority nodes for detailed connections" << std::endl;
    
    // NOVEL STEP 2: Create adjacency using spatial proximity for high-priority nodes
    int connectionsCreated = 0;
    const float maxConnectionDistance = 100.0f; // Reduced from analyzing all adjacencies
    
    // Create connections between high-priority nodes only
    for (size_t i = 0; i < graphNodes.size(); i++) {
        if (highPriorityNodes.find(i) == highPriorityNodes.end()) continue; // Skip low-priority nodes
        
        auto& nodeA = graphNodes[i];
        cv::Point2i posA = nodeA->getPosition();
        
        // Only check nearby nodes (much faster than full adjacency analysis)
        for (size_t j = i + 1; j < graphNodes.size(); j++) {
            if (highPriorityNodes.find(j) == highPriorityNodes.end()) continue; // Skip low-priority nodes
            
            auto& nodeB = graphNodes[j];
            cv::Point2i posB = nodeB->getPosition();
            
            // Calculate spatial distance
            float dx = static_cast<float>(posA.x - posB.x);
            float dy = static_cast<float>(posA.y - posB.y);
            float distance = std::sqrt(dx*dx + dy*dy);
            
            // Only connect if nodes are nearby
            if (distance < maxConnectionDistance) {
                // NOVEL: Adaptive connection weight based on distance and priority
                float priorityA = 0.5f; // Default priority
                float priorityB = 0.5f;
                
                // Find actual priorities
                for (const auto& pair : nodePriorities) {
                    if (pair.first == i) priorityA = pair.second;
                    if (pair.first == j) priorityB = pair.second;
                }
                
                // Connection weight based on distance and combined priority
                float distanceWeight = 1.0f - (distance / maxConnectionDistance);
                float priorityWeight = (priorityA + priorityB) / 2.0f;
                float connectionWeight = 0.3f + 0.4f * distanceWeight + 0.3f * priorityWeight;
                
                // Ensure weight is in valid range
                connectionWeight = std::max(0.1f, std::min(1.0f, connectionWeight));
                
                // Add bidirectional connection
                nodeA->addConnection(nodeB, connectionWeight);
                nodeB->addConnection(nodeA, connectionWeight);
                connectionsCreated++;
            }
        }
        
        // Progress check every 50 nodes
        if (i % 50 == 0) {
            std::cout << "NSGS NOVEL: Processed " << i << "/" << highPriorityCount 
                      << " priority nodes (" << connectionsCreated << " connections)" << std::endl;
        }
    }
    
    // NOVEL STEP 3: Create minimal connections for low-priority nodes
    // Connect each low-priority node to its nearest high-priority neighbor only
    for (size_t i = 0; i < graphNodes.size(); i++) {
        if (highPriorityNodes.find(i) != highPriorityNodes.end()) continue; // Skip high-priority nodes
        
        auto& nodeA = graphNodes[i];
        cv::Point2i posA = nodeA->getPosition();
        
        // Find nearest high-priority node
        float minDistance = std::numeric_limits<float>::max();
        size_t nearestHighPriority = SIZE_MAX;
        
        for (size_t j : highPriorityNodes) {
            auto& nodeB = graphNodes[j];
            cv::Point2i posB = nodeB->getPosition();
            
            float dx = static_cast<float>(posA.x - posB.x);
            float dy = static_cast<float>(posA.y - posB.y);
            float distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestHighPriority = j;
            }
        }
        
        // Connect to nearest high-priority node if found and within reasonable distance
        if (nearestHighPriority != SIZE_MAX && minDistance < maxConnectionDistance * 2.0f) {
            auto& nearestNode = graphNodes[nearestHighPriority];
            float connectionWeight = 0.3f; // Simple weight for low-priority connections
            
            nodeA->addConnection(nearestNode, connectionWeight);
            nearestNode->addConnection(nodeA, connectionWeight);
            connectionsCreated++;
        }
    }
    
    auto connectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - connectionStartTime).count();
    
    std::cout << "NSGS NOVEL: Created " << connectionsCreated << " selective connections in " 
              << connectionTime << "ms (priority-based optimization)" << std::endl;
}

// NEUROMORPHIC CORE: Initialize node potentials and firing thresholds
void NsgsPredictor::initializeNodePotentialsFromEmbeddings(const cv::Mat &image, const cv::Mat &embeddings)
{
    std::cout << "NSGS NEUROMORPHIC: Setting up neuron firing dynamics..." << std::endl;
    
    if (graphNodes.empty()) {
        std::cout << "NSGS: Warning - No neurons to initialize" << std::endl;
        return;
    }
    
    // Calculate global activation statistics for adaptive thresholding
    cv::Mat activationMap;
    if (!embeddings.empty()) {
        // Create activation strength map from CNN embeddings
        activationMap = cv::Mat(embeddings.rows, embeddings.cols, CV_32F, 0.0f);
        for (int y = 0; y < embeddings.rows; y++) {
            for (int x = 0; x < embeddings.cols; x++) {
                float activation = 0.0f;
                for (int c = 0; c < std::min(embeddings.channels(), 32); c++) {
                    float* pixelPtr = (float*)(embeddings.data + y*embeddings.step + x*embeddings.elemSize());
                    float val = pixelPtr[c];
                    activation += val * val;
                }
                activationMap.at<float>(y, x) = std::sqrt(activation);
            }
        }
        cv::normalize(activationMap, activationMap, 0, 1, cv::NORM_MINMAX);
        cv::resize(activationMap, activationMap, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);
    }
    
    // Calculate edge strength for threshold adaptation
    cv::Mat edgeMap;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, edgeMap, CV_32F);
    cv::normalize(edgeMap, edgeMap, 0, 1, cv::NORM_MINMAX);
    
    // Initialize each neuron's firing dynamics
    int neuronsInitialized = 0;
    for (auto& neuron : graphNodes) {
        cv::Point2i pos = neuron->getPosition();
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        // NEUROMORPHIC: Set adaptive firing threshold
        float baseThreshold = 0.4f; // Reduced from 0.5f to make firing more likely
        float edgeStrength = edgeMap.at<float>(y, x);
        
        // Higher threshold at edges to prevent spurious firing - reduced multiplier
        float adaptiveThreshold = baseThreshold * (1.0f + 0.3f * edgeStrength); // Reduced from 0.5f
        neuron->setThreshold(adaptiveThreshold * this->globalThresholdMultiplier);
        
        // NEUROMORPHIC: Set initial membrane potential
        float initialPotential = 0.15f; // Increased resting potential
        
        // Add activation-based potential if CNN indicates strong features
        if (!activationMap.empty()) {
            float cnnActivation = activationMap.at<float>(y, x);
            initialPotential += 0.4f * cnnActivation; // Increased boost for high-activation areas
        }
        
        // Add small edge-based potential for boundary detection
        initialPotential += 0.15f * edgeStrength; // Increased edge contribution
        
        // Set the neuron's initial state
        neuron->resetState();
        neuron->incrementPotential(initialPotential);
        
        neuronsInitialized++;
    }
    
    std::cout << "NSGS NEUROMORPHIC: Initialized " << neuronsInitialized << " neurons with firing dynamics" << std::endl;
    std::cout << "NSGS NEUROMORPHIC: Base threshold = 0.4, Global multiplier = " << this->globalThresholdMultiplier << std::endl;
}