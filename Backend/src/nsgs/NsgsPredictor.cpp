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
    // This aligns with the paper: "We use SLIC superpixels as the basis for our PE graph"
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
        
        // Create SLIC segmentation
        int numSuperpixels = std::min(1000, width * height / 100); // Adaptive number based on image size
        int numIterations = 10;
        float ruler = 10.0f; // Controls compactness
        
        // Create and run the SLIC algorithm
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = 
            cv::ximgproc::createSuperpixelSLIC(labImage, cv::ximgproc::SLIC, ruler, numSuperpixels);
        
        slic->iterate(numIterations);
        
        // Get the labels and contours
        slic->getLabels(labels);
        slic->enforceLabelConnectivity();
        
        // Get actual number of superpixels
        int numLabels = slic->getNumberOfSuperpixels();
        std::cout << "NSGS: Created " << numLabels << " SLIC superpixels" << std::endl;
        
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
                centers[label].x += x;
                centers[label].y += y;
                counts[label]++;
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
    
    // Fallback to watershed segmentation if SLIC failed
    if (!useSuperpixels) {
        // Convert to grayscale
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        
        // Apply Gaussian blur to reduce noise
        cv::GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0);
        
        // Apply threshold to get binary image
        cv::Mat binaryImage;
        cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        
        // Perform morphological operations to enhance watershed
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
        
        // Sure background
        cv::Mat sure_bg;
        cv::dilate(binaryImage, sure_bg, kernel, cv::Point(-1, -1), 3);
        
        // Distance transform for finding sure foreground
        cv::Mat dist_transform;
        cv::distanceTransform(binaryImage, dist_transform, cv::DIST_L2, 5);
        
        // Sure foreground
        cv::Mat sure_fg;
        double minVal, maxVal;
        cv::minMaxLoc(dist_transform, &minVal, &maxVal);
        cv::threshold(dist_transform, sure_fg, 0.6 * maxVal, 255, cv::THRESH_BINARY);
        sure_fg.convertTo(sure_fg, CV_8U);
        
        // Unknown region
        cv::Mat unknown;
        cv::subtract(sure_bg, sure_fg, unknown);
        
        // Markers for watershed
        cv::Mat markers = cv::Mat::zeros(grayImage.size(), CV_32S);
        cv::connectedComponents(sure_fg, markers);
        markers = markers + 1;
        markers.setTo(0, unknown);
        
        // Apply watershed
        cv::watershed(image, markers);
        markers.convertTo(labels, CV_32S);
        
        // Count unique labels and find centers
        std::unordered_map<int, std::vector<cv::Point>> labelPoints;
        int maxLabel = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = labels.at<int>(y, x);
                if (label > 0) { // Ignore watershed boundaries (-1)
                    labelPoints[label].push_back(cv::Point(x, y));
                    maxLabel = std::max(maxLabel, label);
                }
            }
        }
        
        // Calculate centers of regions
        centers.resize(maxLabel + 1);
        for (int i = 1; i <= maxLabel; i++) {
            if (!labelPoints[i].empty()) {
                cv::Point2d center(0, 0);
                for (const auto& p : labelPoints[i]) {
                    center.x += p.x;
                    center.y += p.y;
                }
                center.x /= labelPoints[i].size();
                center.y /= labelPoints[i].size();
                centers[i] = center;
            }
        }
        
        std::cout << "NSGS: Created " << maxLabel << " watershed regions" << std::endl;
    }
    
    // Create nodes at superpixel/region centers
    int nodeId = 0;
    for (const auto& center : centers) {
        if (center.x > 0 || center.y > 0) { // Skip empty centers
            auto node = std::make_shared<NeuronNode>(nodeId++, cv::Point2i(center.x, center.y), spikeQueue);
            graphNodes.push_back(node);
        }
    }
    
    std::cout << "NSGS: Created " << graphNodes.size() << " graph nodes from segmentation" << std::endl;
    
    // Extract rich features for each node
    extractNodeFeatures(image, embeddings);
    
    // Register edge strengths for priority scheduling
    registerNodeEdgeStrengths(image);
    
    // Create connections between nodes based on region adjacency
    // We'll consider two regions adjacent if they share a boundary
    std::vector<std::unordered_set<int>> adjacencyList(centers.size());
    
    // Find adjacency relationships by checking neighborhood of each pixel
    // Using a more efficient approach - only check right and bottom neighbors
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            int currentLabel = labels.at<int>(y, x);
            if (currentLabel <= 0 || currentLabel >= static_cast<int>(centers.size())) continue; // Skip invalid labels
            
            // Check right and bottom neighbors only (left and top will be covered in other iterations)
            int rightLabel = labels.at<int>(y, x + 1);
            int downLabel = labels.at<int>(y + 1, x);
            
            // If label changes, regions are adjacent
            if (currentLabel != rightLabel && rightLabel > 0 && rightLabel < static_cast<int>(centers.size())) {
                adjacencyList[currentLabel].insert(rightLabel);
                adjacencyList[rightLabel].insert(currentLabel);
            }
            
            if (currentLabel != downLabel && downLabel > 0 && downLabel < static_cast<int>(centers.size())) {
                adjacencyList[currentLabel].insert(downLabel);
                adjacencyList[downLabel].insert(currentLabel);
            }
        }
    }
    
    // No need to remove duplicates since we're using std::unordered_set
    
    // Instead of a full matrix for processed flag, use a set of processed pairs
    std::set<std::pair<int, int>> processedPairs;
    
    // Create connections between adjacent regions' nodes
    std::cout << "NSGS: Creating connections between " << graphNodes.size() << " nodes..." << std::endl;
    auto connectionStartTime = std::chrono::high_resolution_clock::now();
    auto connectionTimeout = connectionStartTime + std::chrono::seconds(10); // 10 second timeout
    
    int connectionsCreated = 0;
    int nodesProcessed = 0;
    
    for (size_t i = 0; i < graphNodes.size(); i++) {
        // Progress reporting and timeout check every 50 nodes
        if (i % 50 == 0) {
            if (std::chrono::high_resolution_clock::now() > connectionTimeout) {
                std::cout << "NSGS: Connection creation timeout after processing " << nodesProcessed 
                          << " nodes with " << connectionsCreated << " connections" << std::endl;
                break;
            }
            std::cout << "NSGS: Processing node " << i << "/" << graphNodes.size() 
                      << " (" << connectionsCreated << " connections)" << std::endl;
        }
        
        if (i >= adjacencyList.size()) continue;
        
        const std::vector<float>& nodeFeatures = graphNodes[i]->getFeatures();
        cv::Point2i nodePos = graphNodes[i]->getPosition();
        
        for (int neighborLabel : adjacencyList[i]) {
            if (neighborLabel >= static_cast<int>(graphNodes.size()) || neighborLabel < 0) continue;
            
            // Create ordered pair for tracking
            std::pair<int, int> nodePair(std::min(static_cast<int>(i), neighborLabel), 
                                         std::max(static_cast<int>(i), neighborLabel));
            
            // Skip if already processed this pair (symmetric connection)
            if (processedPairs.find(nodePair) != processedPairs.end()) continue;
            processedPairs.insert(nodePair);
            
            auto& neighborNode = graphNodes[neighborLabel];
            const std::vector<float>& neighborFeatures = neighborNode->getFeatures();
            cv::Point2i neighborPos = neighborNode->getPosition();
            
            // SIMPLIFIED feature similarity calculation for performance
            float connectionWeight = 0.5f; // Default weight
            
            // Calculate basic similarity only if features are available
            if (!nodeFeatures.empty() && !neighborFeatures.empty()) {
                size_t minSize = std::min(nodeFeatures.size(), neighborFeatures.size());
                
                // Simple dot product similarity (much faster than multiple calculations)
                float dotProduct = 0.0f;
                float normA = 0.0f;
                float normB = 0.0f;
                
                // Use only first 10 features for speed (color + some texture)
                size_t maxFeatures = std::min(size_t(10), minSize);
                for (size_t k = 0; k < maxFeatures; k++) {
                    float a = nodeFeatures[k];
                    float b = neighborFeatures[k];
                    dotProduct += a * b;
                    normA += a * a;
                    normB += b * b;
                }
                
                // Calculate cosine similarity
                if (normA > 0 && normB > 0) {
                    float similarity = dotProduct / (std::sqrt(normA) * std::sqrt(normB));
                    similarity = std::max(0.0f, std::min(1.0f, similarity)); // Clamp to [0,1]
                    connectionWeight = 0.3f + 0.7f * similarity;
                }
            }
            
            // Simple spatial distance modulation
            float dx = static_cast<float>(nodePos.x - neighborPos.x);
            float dy = static_cast<float>(nodePos.y - neighborPos.y);
            float spatialDistance = std::sqrt(dx*dx + dy*dy);
            
            // Weaken connection if nodes are far apart
            if (spatialDistance > 50.0f) {
                connectionWeight *= 0.8f;
            }
            
            // Ensure weight stays in valid range
            connectionWeight = std::max(0.1f, std::min(1.0f, connectionWeight));
            
            // Add connection with the calculated weight (symmetric)
            graphNodes[i]->addConnection(neighborNode, connectionWeight);
            neighborNode->addConnection(graphNodes[i], connectionWeight);
            connectionsCreated++;
        }
        nodesProcessed++;
    }
    
    auto connectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - connectionStartTime).count();
    
    std::cout << "NSGS: Created " << connectionsCreated << " connections in " 
              << connectionTime << "ms (" << nodesProcessed << " nodes processed)" << std::endl;
}

void NsgsPredictor::initializeNodePotentials(const cv::Mat &embeddings)
{
    // Process actual embeddings from the model's output tensor
    // These embeddings contain rich semantic information about each position
    
    if (embeddings.empty()) {
        std::cout << "NSGS: Warning - Empty embeddings provided" << std::endl;
        return;
    }
    
    // Extract embedding dimensions
    int embeddingHeight = embeddings.rows;
    int embeddingWidth = embeddings.cols;
    int embeddingChannels = embeddings.channels();
    
    std::cout << "NSGS: Processing embeddings of size " << embeddingWidth << "x" 
              << embeddingHeight << " with " << embeddingChannels << " channels" << std::endl;
    
    // Scaling factors for mapping between embedding resolution and node positions
    float scaleX = static_cast<float>(embeddingWidth) / static_cast<float>((int)this->inputShapes[0][3]);
    float scaleY = static_cast<float>(embeddingHeight) / static_cast<float>((int)this->inputShapes[0][2]);
    
    for (auto &node : graphNodes) {
        // Reset node state
        node->resetState();
        
        // Calculate corresponding position in embedding space
        cv::Point2i pos = node->getPosition();
        int embX = std::min(embeddingWidth - 1, std::max(0, static_cast<int>(pos.x * scaleX)));
        int embY = std::min(embeddingHeight - 1, std::max(0, static_cast<int>(pos.y * scaleY)));
        
        // Set adaptive threshold based on local edge information
        // Calculate local gradient magnitude - higher in boundary areas
        float gradX = 0.0f, gradY = 0.0f;
        if (embX > 0 && embX < embeddingWidth - 1 && embY > 0 && embY < embeddingHeight - 1) {
            // Calculate gradient using multiple channels for robustness
            for (int c = 0; c < std::min(3, embeddingChannels); c++) {
                float right = 0, left = 0, down = 0, up = 0;
                
                if (embeddingChannels == 1) {
                    right = embeddings.at<float>(embY, embX + 1);
                    left = embeddings.at<float>(embY, embX - 1);
                    down = embeddings.at<float>(embY + 1, embX);
                    up = embeddings.at<float>(embY - 1, embX);
                } else {
                    // Fix: Safe multi-channel access with proper checks for channel count
                    float* pixelPtr;
                    
                    // Right pixel
                    pixelPtr = (float*)(embeddings.data + embY*embeddings.step + (embX+1)*embeddings.elemSize());
                    right = pixelPtr[c];
                    
                    // Left pixel
                    pixelPtr = (float*)(embeddings.data + embY*embeddings.step + (embX-1)*embeddings.elemSize());
                    left = pixelPtr[c];
                    
                    // Down pixel
                    pixelPtr = (float*)(embeddings.data + (embY+1)*embeddings.step + embX*embeddings.elemSize());
                    down = pixelPtr[c];
                    
                    // Up pixel
                    pixelPtr = (float*)(embeddings.data + (embY-1)*embeddings.step + embX*embeddings.elemSize());
                    up = pixelPtr[c];
                }
                
                gradX += (right - left) / 2.0f;
                gradY += (down - up) / 2.0f;
            }
        }
        
        // Normalize and use gradient for threshold adaptation
        float gradientMagnitude = std::sqrt(gradX * gradX + gradY * gradY);
        float normalizedGradient = std::min(1.0f, gradientMagnitude / 2.0f);
        
        // Higher threshold in boundary areas (high gradient) to prevent unwanted class propagation
        float baseThreshold = 0.5f;
        float adaptiveThreshold = baseThreshold * (1.0f + normalizedGradient);
        node->setThreshold(adaptiveThreshold);
        
        // Extract feature vector from embeddings for this node
        std::vector<float> featureVector;
        featureVector.reserve(embeddingChannels);
        
        // For single channel embeddings
        if (embeddingChannels == 1) {
            // Extract a small local neighborhood to capture local context
            const int patchRadius = 2;
            for (int dy = -patchRadius; dy <= patchRadius; dy++) {
                for (int dx = -patchRadius; dx <= patchRadius; dx++) {
                    int nx = std::min(embeddingWidth - 1, std::max(0, embX + dx));
                    int ny = std::min(embeddingHeight - 1, std::max(0, embY + dy));
                    featureVector.push_back(embeddings.at<float>(ny, nx));
                }
            }
        } 
        // For multi-channel embeddings (like prototypes from YOLOv8-seg models)
        else {
            // Direct feature extraction from multi-channel data (common in CNN embeddings)
            for (int c = 0; c < embeddingChannels; c++) {
                featureVector.push_back(embeddings.at<cv::Vec<float, 32>>(embY, embX)[c]);
            }
        }
        
        // Assign features to node
        node->setFeatures(featureVector);
        
        // Initial potential slightly higher in higher gradient areas
        // This biases early activity toward boundary regions which helps segmentation
        float initialPotential = 0.05f + 0.15f * normalizedGradient;
        node->incrementPotential(initialPotential);
    }
    
    std::cout << "NSGS: Node potentials and features initialized from model embeddings" << std::endl;
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
    std::cout << "NSGS: Starting spike propagation with timeout safeguards..." << std::endl;
    
    // Start timeout counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto timeout = startTime + std::chrono::seconds(7); // 7-second timeout
    
    // Content-aware initial spike selection instead of random seeding
    // This follows the paper: "Identify initial active PEs (e.g., high contrast regions)"
    
    // First, collect node characteristics for informed selection
    std::vector<std::pair<size_t, float>> nodeScores;
    
    // Safety check
    if (graphNodes.empty()) {
        std::cerr << "NSGS: No nodes available for spike propagation" << std::endl;
        return;
    }
    
    nodeScores.reserve(graphNodes.size());
    
    // Get image dimensions for normalization
    int width = (int)this->inputShapes[0][3];
    int height = (int)this->inputShapes[0][2];
    
    // Check for timeout before creating activations
    if (std::chrono::high_resolution_clock::now() > timeout) {
        std::cout << "NSGS: Timeout before activation setup, skipping propagation" << std::endl;
        return;
    }
    
    // Extract CNN activations if available
    cv::Mat cnnActivations;
    if (this->hasMask && outputTensors.size() > 1) {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        if (!maskOutput) {
            std::cerr << "NSGS: Invalid mask output tensor data" << std::endl;
        } else {
            std::vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
            
            // Proper bounds checking
            if (maskShape.size() >= 4) {
                std::cout << "NSGS: Processing mask activations..." << std::endl;
                // Create activation magnitude map by summing across channels
                int protoChannels = maskShape[1];
                int protoHeight = maskShape[2];
                int protoWidth = maskShape[3];
                
                // Calculate activation magnitude (L2 norm across channels)
                cv::Mat embeddings(protoHeight, protoWidth, CV_32FC(protoChannels), maskOutput);
                cnnActivations = cv::Mat(protoHeight, protoWidth, CV_32F, 0.0f);
                
                // Check for timeout before processing embeddings
                if (std::chrono::high_resolution_clock::now() > timeout) {
                    std::cout << "NSGS: Timeout before embeddings processing, skipping" << std::endl;
                    return;
                }
                
                // Process embeddings with periodic timeout checks
                int processed_rows = 0;
                for (int y = 0; y < protoHeight; y++) {
                    if (y % 10 == 0 && std::chrono::high_resolution_clock::now() > timeout) {
                        std::cout << "NSGS: Timeout during embeddings processing, using partial data" << std::endl;
                        break;
                    }
                    
                    for (int x = 0; x < protoWidth; x++) {
                        float sum = 0.0f;
                        // Safe access using direct pointer arithmetic
                        float* pixelPtr = (float*)(embeddings.data + y*embeddings.step + x*embeddings.elemSize());
                        for (int c = 0; c < std::min(protoChannels, 32); c++) {
                            float val = pixelPtr[c];
                            sum += val * val;
                        }
                        cnnActivations.at<float>(y, x) = std::sqrt(sum);
                    }
                    processed_rows++;
                }
                
                std::cout << "NSGS: Processed " << processed_rows << " of " << protoHeight << " embedding rows" << std::endl;
                
                // Normalize activations to [0,1] range
                double minVal, maxVal;
                cv::minMaxLoc(cnnActivations, &minVal, &maxVal);
                if (maxVal > minVal) {
                    cnnActivations = (cnnActivations - minVal) / (maxVal - minVal);
                }
                
                // Resize to match input dimensions
                cv::resize(cnnActivations, cnnActivations, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            }
        }
    }
    
    // Check for timeout before calculating node scores
    if (std::chrono::high_resolution_clock::now() > timeout) {
        std::cout << "NSGS: Timeout before node scoring, skipping propagation" << std::endl;
        return;
    }
    
    std::cout << "NSGS: Calculating node scores..." << std::endl;
    
    // Calculate score for each node based on multiple criteria
    size_t nodesProcessed = 0;
    for (size_t i = 0; i < graphNodes.size(); i++) {
        // Check for timeout every 100 nodes
        if (i % 100 == 0 && std::chrono::high_resolution_clock::now() > timeout) {
            std::cout << "NSGS: Timeout during node scoring, using partial scores" << std::endl;
            break;
        }
        
        auto &node = graphNodes[i];
        const std::vector<float> &features = node->getFeatures();
        cv::Point2i pos = node->getPosition();
        
        float score = 0.0f;
        
        // 1. Feature-based metrics (use gradient magnitude for edge detection)
        // We're looking for high-contrast areas as mentioned in the paper
        if (!features.empty()) {
            // Extract gradient magnitude features (assuming features layout from extractNodeFeatures)
            // Gradient features typically begin at index 19 (after color and texture features)
            if (features.size() > 19) {
                float gradientMagnitude = 0.0f;
                for (size_t j = 19; j < std::min(size_t(27), features.size()); j++) {
                    gradientMagnitude += features[j];
                }
                gradientMagnitude /= 8.0f; // Normalize (8 gradient orientation bins)
                
                // High gradient magnitude (edges) gets high score
                score += 0.4f * gradientMagnitude;
            }
            
            // 2. Texture complexity (from LBP features)
            // Complex texture regions are often important for segmentation
            if (features.size() > 3) {
                float textureComplexity = 0.0f;
                for (size_t j = 3; j < std::min(size_t(19), features.size()); j++) {
                    textureComplexity += features[j] * features[j]; // Sum of squared LBP histogram values
                }
                textureComplexity = std::sqrt(textureComplexity);
                
                // Higher texture complexity gets higher score
                score += 0.2f * textureComplexity;
            }
        }
        
        // 3. CNN activation-based score (if available)
        if (!cnnActivations.empty()) {
            // Get corresponding position in activation map
            int x = std::min(width - 1, std::max(0, pos.x));
            int y = std::min(height - 1, std::max(0, pos.y));
            
            // Extract activation value at this position
            float activation = cnnActivations.at<float>(y, x);
            
            // Strong CNN activations get high score (this is important in object regions)
            score += 0.4f * activation;
        }
        
        // 4. Spatial distribution bonus (helps spread initial activations)
        // Normalize position to [0,1] range
        float normalizedX = static_cast<float>(pos.x) / width;
        float normalizedY = static_cast<float>(pos.y) / height;
        
        // Distance from center (center gets low bonus, periphery gets high bonus)
        float centerDistX = std::abs(normalizedX - 0.5f) * 2.0f; // [0,1] range
        float centerDistY = std::abs(normalizedY - 0.5f) * 2.0f; // [0,1] range
        float centerDist = std::sqrt(centerDistX*centerDistX + centerDistY*centerDistY) / std::sqrt(2.0f);
        
        // Add small bonus for spatial diversity
        score += 0.1f * centerDist;
        
        // Store node index and its score
        nodeScores.push_back(std::make_pair(i, score));
        nodesProcessed++;
    }
    
    std::cout << "NSGS: Processed " << nodesProcessed << " of " << graphNodes.size() << " nodes for scoring" << std::endl;
    
    // Check for timeout before sorting
    if (std::chrono::high_resolution_clock::now() > timeout) {
        std::cout << "NSGS: Timeout before score sorting, skipping propagation" << std::endl;
        return;
    }
    
    std::cout << "NSGS: Sorting node scores..." << std::endl;
    
    // Sort nodes by score in descending order
    std::sort(nodeScores.begin(), nodeScores.end(), 
              [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b) {
                  return a.second > b.second;
              });
    
    // Select top N nodes with highest scores for initial activation
    // Calculate adaptive number of initial spikes based on image complexity and node count
    int minSpikes = 5;
    int maxSpikes = 50;
    int adaptiveCount = static_cast<int>(0.01f * graphNodes.size()); // 1% of nodes
    int numInitialSpikes = std::min(maxSpikes, std::max(minSpikes, adaptiveCount));
    
    std::cout << "NSGS: Activating " << numInitialSpikes << " initial nodes based on content" << std::endl;
    
    // Also keep track of activated positions for visualization
    std::vector<cv::Point2i> activatedPositions;
    
    // Check for timeout before activation
    if (std::chrono::high_resolution_clock::now() > timeout) {
        std::cout << "NSGS: Timeout before node activation, skipping propagation" << std::endl;
        return;
    }
    
    // Activate nodes with highest scores
    for (int i = 0; i < std::min(numInitialSpikes, static_cast<int>(nodeScores.size())); i++) {
        size_t nodeIdx = nodeScores[i].first;
        float score = nodeScores[i].second;
        
        // Scale activation potential by score (higher score = stronger activation)
        float activationStrength = 0.5f + 0.5f * score;
        graphNodes[nodeIdx]->incrementPotential(activationStrength);
        graphNodes[nodeIdx]->checkAndFire(); // Check if this causes a spike
        
        // Store position for logging
        activatedPositions.push_back(graphNodes[nodeIdx]->getPosition());
    }
    
    // Add a smaller number of random activations for diversity
    const int numRandomSpikes = numInitialSpikes / 5; // 20% of the total
    if (!graphNodes.empty()) {
        std::cout << "NSGS: Adding " << numRandomSpikes << " random activations for diversity" << std::endl;
        for (int i = 0; i < numRandomSpikes; i++) {
            int randomIdx = rand() % graphNodes.size();
            graphNodes[randomIdx]->incrementPotential(0.6f); // Moderate activation
            graphNodes[randomIdx]->checkAndFire();
        }
    }
    
    // Log statistics of initial activations
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startTime).count();
    std::cout << "NSGS: Node activation completed in " << elapsedTime << "ms" << std::endl;
    std::cout << "NSGS: Created connections with feature-based weights (spike propagation ready)" << std::endl;
}

cv::Mat NsgsPredictor::reconstructFromNeuralGraph()
{
    // Implementation moved to reconstructFromNeuralGraph.cpp
    // This method is kept as a stub to maintain compatibility
    // See that file for the enhanced implementation using spatial clustering
    
    // Safety checks before fallback implementation
    if (inputShapes.empty() || inputShapes[0].size() < 4) {
        std::cerr << "NSGS: Invalid input shapes for reconstruction" << std::endl;
        return cv::Mat();
    }
    
    // Fallback implementation in case main implementation unavailable
    int width = (int)this->inputShapes[0][3];
    int height = (int)this->inputShapes[0][2];
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    
    // For each node, color its area based on its class ID
    for (auto &node : graphNodes) {
        if (!node) continue; // Skip null nodes
        
        int classId = node->getClassId();
        if (classId >= 0) { // If node has been assigned a class
            cv::Point2i pos = node->getPosition();
            
            // Ensure position is within image bounds
            if (pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height) {
                int radius = 4; // Radius of influence for each node
                
                // Draw a filled circle at the node's position with its class ID as the value
                cv::circle(mask, pos, radius, cv::Scalar(classId + 1), -1); // +1 to avoid 0 (background)
            }
        }
    }
    
    return mask;
}

void NsgsPredictor::runAsyncEventProcessing()
{
    // Start timeout counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto timeout = startTime + std::chrono::seconds(10); // 10-second timeout
    
    if (!spikeQueue) {
        std::cerr << "NSGS: SpikeQueue not initialized" << std::endl;
        return;
    }
    
    // Start processing in the queue's thread pool
    spikeQueue->startProcessing(&graphNodes);
    
    // While the queue is processing, periodically check if we need to adapt
    while (spikeQueue->isProcessing()) {
        // Check for timeout
        if (std::chrono::high_resolution_clock::now() > timeout) {
            std::cout << "NSGS: Timeout during spike processing after 10 seconds, continuing with partial results" << std::endl;
            spikeQueue->stopProcessing();
            break;
        }
        
        // Give the threads some time to work (process spikes)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Log progress
        if ((spikeQueue->getProcessedCount() % 100) == 0 && spikeQueue->getProcessedCount() > 0) {
            std::cout << "NSGS: Processed " << spikeQueue->getProcessedCount() 
                      << " spikes, " << spikeQueue->getStateChanges() << " state changes, "
                      << spikeQueue->size() << " in queue" << std::endl;
        }
    }
    
    // Print final statistics
    std::cout << "NSGS: Event processing complete - " 
              << spikeQueue->getProcessedCount() << " spikes processed, "
              << spikeQueue->getStateChanges() << " state changes" << std::endl;
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
    cv::magnitude(gradX, gradY, gradMag);
    
    // Normalize gradient magnitude to [0,1] range for easier prioritization
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
            auto graphTimeout = graphStartTime + std::chrono::seconds(5); // Reduced from 15 seconds to 5 seconds
            
            // Extract features and create graph
            std::cout << "NSGS: Creating neural graph for segmentation..." << std::endl;
            buildSegmentationGraph(image);
            
            // Check timeout after graph creation
            if (std::chrono::high_resolution_clock::now() > graphTimeout) {
                std::cout << "NSGS: Timeout after graph creation, skipping remaining steps but ensuring results are saved" << std::endl;
                delete[] blob;
                return results;  // Return basic results
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
            
            // Check timeout after feature extraction
            if (std::chrono::high_resolution_clock::now() > graphTimeout) {
                std::cout << "NSGS: Timeout after feature extraction, skipping remaining steps but ensuring results are saved" << std::endl;
                delete[] blob;
                return results;  // Return basic results
            }
            
            // Set up timing for propagation
            auto propagationStartTime = std::chrono::high_resolution_clock::now();
            auto propagationTimeout = propagationStartTime + std::chrono::seconds(3); // Reduced from 7 seconds to 3 seconds
            
            // Try to run the propagation with timeout protection
            try {
                // Run propagation on this thread with timeout
                std::cout << "NSGS: Running spike propagation with timeout protection..." << std::endl;
                this->propagateSpikes(true);
                
                // Check if timeout occurred
                if (std::chrono::high_resolution_clock::now() > propagationTimeout) {
                    std::cout << "NSGS: Timeout during propagation, proceeding with partial results" << std::endl;
                } else {
                    std::cout << "NSGS: Propagation completed successfully" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "NSGS: Exception during spike propagation: " << e.what() << std::endl;
            }
            
            // Always proceed with event processing regardless of propagation completion
            std::cout << "NSGS: Running event processing..." << std::endl;
            this->runAsyncEventProcessing();
            
            // Reconstruct segmentation from neural graph
            std::cout << "NSGS: Reconstructing segmentation from neural graph..." << std::endl;
            cv::Mat segMask = this->reconstructFromNeuralGraph();
            
            // If segmentation was successful, update masks in results
            if (!segMask.empty()) {
                std::cout << "NSGS: Updating results with graph-based segmentation" << std::endl;
                for (auto& result : results) {
                    // Determine if this detection overlaps with any labeled regions in the segmentation
                    cv::Rect bbox = result.box;
                    // Ensure bounds are valid
                    int x = std::max(0, bbox.x);
                    int y = std::max(0, bbox.y);
                    int width = std::min(segMask.cols - x, bbox.width);
                    int height = std::min(segMask.rows - y, bbox.height);
                    
                    if (width <= 0 || height <= 0) continue; // Skip invalid regions
                    
                    cv::Rect validRect(x, y, width, height);
                    cv::Mat maskROI = segMask(validRect);
                    
                    // If there's a valid segmentation for this detection's region
                    if (!maskROI.empty() && cv::countNonZero(maskROI) > 0) {
                        // Create new mask for this detection
                        cv::Mat newMask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
                        
                        // Copy the segmentation mask for this region
                        cv::Mat segROI = newMask(bbox);
                        maskROI.copyTo(segROI);
                        
                        // Update the result with this new mask
                        result.boxMask = newMask;
                    }
                }
                fullProcessingCompleted = true;
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