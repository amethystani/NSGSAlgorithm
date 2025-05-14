#include "NsgsPredictor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map> // For watershed region tracking
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
    
    // Get embeddings from model output for feature extraction
    cv::Mat embeddings;
    
    // If we have mask protos from the YOLO segmentation model, use them as embeddings
    if (this->hasMask && outputTensors.size() > 1) {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        std::vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        
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
    std::vector<std::vector<int>> adjacencyList(centers.size(), std::vector<int>());
    
    // Find adjacency relationships by checking neighborhood of each pixel
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int currentLabel = labels.at<int>(y, x);
            
            // Check 4-neighborhood
            int rightLabel = labels.at<int>(y, x + 1);
            int leftLabel = labels.at<int>(y, x - 1);
            int downLabel = labels.at<int>(y + 1, x);
            int upLabel = labels.at<int>(y - 1, x);
            
            // If label changes, regions are adjacent
            if (currentLabel != rightLabel && currentLabel > 0 && rightLabel > 0) {
                adjacencyList[currentLabel].push_back(rightLabel);
                adjacencyList[rightLabel].push_back(currentLabel);
            }
            
            if (currentLabel != downLabel && currentLabel > 0 && downLabel > 0) {
                adjacencyList[currentLabel].push_back(downLabel);
                adjacencyList[downLabel].push_back(currentLabel);
            }
            
            // Don't need to check left and up since they'll be covered in other iterations
        }
    }
    
    // Remove duplicates in adjacency list
    for (auto& neighbors : adjacencyList) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    
    // Create connections between adjacent regions' nodes
    for (size_t i = 0; i < graphNodes.size(); i++) {
        if (i >= adjacencyList.size()) continue;
        
        const std::vector<float>& nodeFeatures = graphNodes[i]->getFeatures();
        cv::Point2i nodePos = graphNodes[i]->getPosition();
        
        for (int neighborLabel : adjacencyList[i]) {
            if (neighborLabel >= static_cast<int>(graphNodes.size())) continue;
            
            auto& neighborNode = graphNodes[neighborLabel];
            const std::vector<float>& neighborFeatures = neighborNode->getFeatures();
            cv::Point2i neighborPos = neighborNode->getPosition();
            
            // Calculate multiple feature similarity metrics and combine them
            float cosineSimilarity = 0.0f;
            float euclideanDistance = 0.0f;
            float spatialDistance = 0.0f;
            float colorSimilarity = 0.0f;
            float textureSimilarity = 0.0f;
            
            // 1. Calculate cosine similarity for overall feature similarity
            if (!nodeFeatures.empty() && !neighborFeatures.empty()) {
                // Find minimum common size
                size_t minSize = std::min(nodeFeatures.size(), neighborFeatures.size());
                
                float dotProduct = 0.0f;
                float normA = 0.0f;
                float normB = 0.0f;
                
                // Calculate dot product and norms
                for (size_t k = 0; k < minSize; k++) {
                    dotProduct += nodeFeatures[k] * neighborFeatures[k];
                    normA += nodeFeatures[k] * nodeFeatures[k];
                    normB += neighborFeatures[k] * neighborFeatures[k];
                }
                
                // Calculate cosine similarity
                if (normA > 0 && normB > 0) {
                    cosineSimilarity = dotProduct / (std::sqrt(normA) * std::sqrt(normB));
                }
                
                // Calculate euclidean distance (then convert to similarity)
                float euclideanDistanceSquared = 0.0f;
                for (size_t k = 0; k < minSize; k++) {
                    float diff = nodeFeatures[k] - neighborFeatures[k];
                    euclideanDistanceSquared += diff * diff;
                }
                euclideanDistance = std::sqrt(euclideanDistanceSquared);
                float euclideanSimilarity = 1.0f / (1.0f + euclideanDistance); // Convert distance to similarity
                
                // 2. Calculate spatial distance similarity
                float dx = static_cast<float>(nodePos.x - neighborPos.x);
                float dy = static_cast<float>(nodePos.y - neighborPos.y);
                spatialDistance = std::sqrt(dx*dx + dy*dy);
                
                // Convert to similarity measure (closer = more similar)
                // Normalize based on image dimensions
                float maxDistance = std::sqrt(static_cast<float>(width*width + height*height));
                float spatialSimilarity = 1.0f - (spatialDistance / maxDistance);
                
                // 3. Calculate separate similarities for color and texture features
                // Assuming the first 3 elements are color (Lab) and next 16 are texture (LBP)
                if (minSize >= 19) { // Make sure we have both color and texture features
                    // Color similarity (first 3 features)
                    float colorDotProduct = 0.0f;
                    float colorNormA = 0.0f, colorNormB = 0.0f;
                    
                    for (size_t k = 0; k < 3; k++) {
                        colorDotProduct += nodeFeatures[k] * neighborFeatures[k];
                        colorNormA += nodeFeatures[k] * nodeFeatures[k];
                        colorNormB += neighborFeatures[k] * neighborFeatures[k];
                    }
                    
                    if (colorNormA > 0 && colorNormB > 0) {
                        colorSimilarity = colorDotProduct / (std::sqrt(colorNormA) * std::sqrt(colorNormB));
                    }
                    
                    // Texture similarity (next 16 features)
                    float textureDotProduct = 0.0f;
                    float textureNormA = 0.0f, textureNormB = 0.0f;
                    
                    for (size_t k = 3; k < 19; k++) {
                        textureDotProduct += nodeFeatures[k] * neighborFeatures[k];
                        textureNormA += nodeFeatures[k] * nodeFeatures[k];
                        textureNormB += neighborFeatures[k] * neighborFeatures[k];
                    }
                    
                    if (textureNormA > 0 && textureNormB > 0) {
                        textureSimilarity = textureDotProduct / (std::sqrt(textureNormA) * std::sqrt(textureNormB));
                    }
                }
            }
            
            // 4. Combine all similarity metrics with appropriate weights
            // Prioritize feature similarity but also consider spatial proximity
            float combinedSimilarity = 0.0f;
            
            size_t minFeatureSize = std::min(nodeFeatures.size(), neighborFeatures.size());
            
            // Use all features with higher weight on color and texture
            if (minFeatureSize >= 19) {
                combinedSimilarity = 0.40f * cosineSimilarity +
                                    0.30f * colorSimilarity +
                                    0.20f * textureSimilarity;
            } else {
                // Fallback to basic similarity if we don't have all features
                combinedSimilarity = 0.70f * cosineSimilarity +
                                    0.30f * colorSimilarity;
            }
            
            // Calculate and apply edge-based feature weights if we have gradient features
            if (minFeatureSize > 19) { // Make sure we have gradient features
                float edgeWeight = 0.0f;
                
                // Extract edge features (typically after the texture features)
                for (size_t j = 19; j < std::min(size_t(27), minFeatureSize); j++) {
                    edgeWeight += std::abs(nodeFeatures[j] - neighborFeatures[j]);
                }
                
                // Normalize the edge weight
                edgeWeight = std::min(1.0f, edgeWeight / 8.0f);
                
                // Adjust the similarity by edge strength
                // Stronger edges -> lower similarity -> creates natural segmentation boundaries
                combinedSimilarity *= (1.0f - 0.5f * edgeWeight);
            }
            
            // Ensure similarity is in [0,1] range
            combinedSimilarity = std::max(0.0f, std::min(1.0f, combinedSimilarity));
            
            // Final connection weight calculation:
            // - Scale similarity to create weight in [0.2, 1.0] range
            // - This ensures all connections have minimum strength
            float connectionWeight = 0.2f + 0.8f * combinedSimilarity;
            
            // 5. Apply NSGS-specific weight modulation based on edge features
            // Weaken connections at strong image edges to help form segment boundaries
            float edgeStrength = 0.0f;
            cv::Point2i midpoint((nodePos.x + neighborPos.x) / 2, (nodePos.y + neighborPos.y) / 2);
            
            // Find edge strength features (gradient magnitudes) - assuming they're stored at specific index
            if (minFeatureSize > 19) { // Make sure we have gradient features
                // For this implementation, assume gradient features start at index 19
                // Use the average of gradient features for edge strength
                float nodeEdge = 0.0f, neighborEdge = 0.0f;
                for (size_t j = 19; j < std::min(size_t(27), minFeatureSize); j++) {
                    nodeEdge += nodeFeatures[j];
                    neighborEdge += neighborFeatures[j];
                }
                
                nodeEdge /= 8.0f; // Normalize (8 gradient orientation bins)
                neighborEdge /= 8.0f;
                
                // Higher edge strength = weaker connection
                edgeStrength = (nodeEdge + neighborEdge) / 2.0f;
                
                // Apply edge-based modulation (stronger edges = weaker connections)
                connectionWeight *= (1.0f - 0.5f * edgeStrength);
            }
            
            // Ensure weight stays in valid range
            connectionWeight = std::max(0.1f, std::min(1.0f, connectionWeight));
            
            // Add connection with the calculated weight
            graphNodes[i]->addConnection(neighborNode, connectionWeight);
        }
    }
    
    std::cout << "NSGS: Created connections with feature-based weights" << std::endl;
    
    // Start async spike processing
    spikeQueue->startProcessing(&graphNodes);
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
                float right, left, down, up;
                
                if (embeddingChannels == 1) {
                    right = embeddings.at<float>(embY, embX + 1);
                    left = embeddings.at<float>(embY, embX - 1);
                    down = embeddings.at<float>(embY + 1, embX);
                    up = embeddings.at<float>(embY - 1, embX);
                } else {
                    right = embeddings.at<cv::Vec<float, 32>>(embY, embX + 1)[c];
                    left = embeddings.at<cv::Vec<float, 32>>(embY, embX - 1)[c];
                    down = embeddings.at<cv::Vec<float, 32>>(embY + 1, embX)[c];
                    up = embeddings.at<cv::Vec<float, 32>>(embY - 1, embX)[c];
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
    
    // Compute texture features using Local Binary Patterns (basic version)
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat lbpImage;
    
    // Simple LBP implementation
    lbpImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    for (int y = 1; y < grayImage.rows - 1; y++) {
        for (int x = 1; x < grayImage.cols - 1; x++) {
            uchar center = grayImage.at<uchar>(y, x);
            unsigned char lbpCode = 0;
            
            // Compare with 8 neighbors in clockwise order
            lbpCode |= (grayImage.at<uchar>(y-1, x) >= center) << 7;
            lbpCode |= (grayImage.at<uchar>(y-1, x+1) >= center) << 6;
            lbpCode |= (grayImage.at<uchar>(y, x+1) >= center) << 5;
            lbpCode |= (grayImage.at<uchar>(y+1, x+1) >= center) << 4;
            lbpCode |= (grayImage.at<uchar>(y+1, x) >= center) << 3;
            lbpCode |= (grayImage.at<uchar>(y+1, x-1) >= center) << 2;
            lbpCode |= (grayImage.at<uchar>(y, x-1) >= center) << 1;
            lbpCode |= (grayImage.at<uchar>(y-1, x-1) >= center) << 0;
            
            lbpImage.at<uchar>(y, x) = lbpCode;
        }
    }
    
    // Create feature vector for each node
    for (auto &node : graphNodes) {
        // Reset any existing features
        std::vector<float> featureVector;
        
        // Get node position
        cv::Point2i pos = node->getPosition();
        int x = std::min(image.cols - 1, std::max(0, pos.x));
        int y = std::min(image.rows - 1, std::max(0, pos.y));
        
        // 1. COLOR FEATURES (Lab color space for perceptual uniformity)
        cv::Mat labImage;
        cv::cvtColor(floatImage, labImage, cv::COLOR_BGR2Lab);
        
        // Extract color in local neighborhood (5x5 patch)
        const int colorPatchRadius = 2;
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
        
        // 2. TEXTURE FEATURES (LBP histogram)
        const int texturePatchRadius = 4;
        std::vector<int> lbpHistogram(16, 0); // Simplified LBP using 16 bins
        
        for (int dy = -texturePatchRadius; dy <= texturePatchRadius; dy++) {
            for (int dx = -texturePatchRadius; dx <= texturePatchRadius; dx++) {
                int nx = std::min(lbpImage.cols - 1, std::max(0, x + dx));
                int ny = std::min(lbpImage.rows - 1, std::max(0, y + dy));
                
                uchar lbpValue = lbpImage.at<uchar>(ny, nx);
                lbpHistogram[lbpValue % 16]++; // Simplified binning
            }
        }
        
        // Normalize histogram and add to features
        int totalLbpPixels = (2*texturePatchRadius+1) * (2*texturePatchRadius+1);
        for (int i = 0; i < 16; i++) {
            featureVector.push_back(static_cast<float>(lbpHistogram[i]) / totalLbpPixels);
        }
        
        // 3. GRADIENT FEATURES
        const int gradientPatchRadius = 3;
        std::vector<float> gradientFeatures(8, 0.0f); // 8 orientation bins
        float totalGradientMag = 0.0f;
        
        for (int dy = -gradientPatchRadius; dy <= gradientPatchRadius; dy++) {
            for (int dx = -gradientPatchRadius; dx <= gradientPatchRadius; dx++) {
                int nx = std::min(gradMag.cols - 1, std::max(0, x + dx));
                int ny = std::min(gradMag.rows - 1, std::max(0, y + dy));
                
                float mag = gradMag.at<float>(ny, nx);
                float orient = gradOrient.at<float>(ny, nx);
                
                // Bin orientation (0-360 degrees into 8 bins)
                int bin = static_cast<int>(orient * 8 / (2 * CV_PI)) % 8;
                gradientFeatures[bin] += mag;
                totalGradientMag += mag;
            }
        }
        
        // Normalize gradient histogram and add to features
        if (totalGradientMag > 0) {
            for (int i = 0; i < 8; i++) {
                featureVector.push_back(gradientFeatures[i] / totalGradientMag);
            }
        } else {
            // If no gradient, add zeros
            featureVector.insert(featureVector.end(), 8, 0.0f);
        }
        
        // 4. CNN EMBEDDINGS (if available)
        if (!embeddings.empty()) {
            // Calculate embedding position (may be at different resolution)
            float scaleX = static_cast<float>(embeddings.cols) / static_cast<float>(image.cols);
            float scaleY = static_cast<float>(embeddings.rows) / static_cast<float>(image.rows);
            
            int embX = std::min(embeddings.cols - 1, std::max(0, static_cast<int>(x * scaleX)));
            int embY = std::min(embeddings.rows - 1, std::max(0, static_cast<int>(y * scaleY)));
            
            // Get embedding features
            int embChannels = embeddings.channels();
            
            if (embChannels == 1) {
                // Single channel embeddings
                const int embPatchRadius = 1;
                for (int dy = -embPatchRadius; dy <= embPatchRadius; dy++) {
                    for (int dx = -embPatchRadius; dx <= embPatchRadius; dx++) {
                        int nx = std::min(embeddings.cols - 1, std::max(0, embX + dx));
                        int ny = std::min(embeddings.rows - 1, std::max(0, embY + dy));
                        featureVector.push_back(embeddings.at<float>(ny, nx));
                    }
                }
            } else {
                // Multi-channel embeddings (like from YOLOv8 mask protos)
                // Use every 4th channel to keep feature vector size reasonable
                for (int c = 0; c < embChannels; c += 4) {
                    if (embChannels <= 32) {
                        featureVector.push_back(embeddings.at<cv::Vec<float, 32>>(embY, embX)[c]);
                    } else {
                        // For different channel counts, do a safe access
                        float* pixelPtr = (float*)(embeddings.data + embY*embeddings.step + embX*embeddings.elemSize());
                        featureVector.push_back(pixelPtr[c]);
                    }
                }
            }
        }
        
        // Normalize the entire feature vector (L2 norm)
        float featureNorm = 0.0f;
        for (float f : featureVector) {
            featureNorm += f * f;
        }
        
        if (featureNorm > 0) {
            featureNorm = std::sqrt(featureNorm);
            for (float &f : featureVector) {
                f /= featureNorm;
            }
        }
        
        // Set features for this node
        node->setFeatures(featureVector);
    }
    
    std::cout << "NSGS: Extracted features with dimension " 
              << (graphNodes.empty() ? 0 : graphNodes[0]->getFeatures().size())
              << " for each node" << std::endl;
}

void NsgsPredictor::propagateSpikes(bool adaptToThermal)
{
    // Content-aware initial spike selection instead of random seeding
    // This follows the paper: "Identify initial active PEs (e.g., high contrast regions)"
    
    // First, collect node characteristics for informed selection
    std::vector<std::pair<size_t, float>> nodeScores;
    nodeScores.reserve(graphNodes.size());
    
    // Get image dimensions for normalization
    int width = (int)this->inputShapes[0][3];
    int height = (int)this->inputShapes[0][2];
    
    // Extract CNN activations if available
    cv::Mat cnnActivations;
    if (this->hasMask && outputTensors.size() > 1) {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        std::vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        
        // Create activation magnitude map by summing across channels
        int protoChannels = maskShape[1];
        int protoHeight = maskShape[2];
        int protoWidth = maskShape[3];
        
        // Calculate activation magnitude (L2 norm across channels)
        cv::Mat embeddings(protoHeight, protoWidth, CV_32FC(protoChannels), maskOutput);
        cnnActivations = cv::Mat(protoHeight, protoWidth, CV_32F, 0.0f);
        
        for (int y = 0; y < protoHeight; y++) {
            for (int x = 0; x < protoWidth; x++) {
                float sum = 0.0f;
                for (int c = 0; c < std::min(protoChannels, 32); c++) {
                    float val = embeddings.at<cv::Vec<float, 32>>(y, x)[c];
                    sum += val * val;
                }
                cnnActivations.at<float>(y, x) = std::sqrt(sum);
            }
        }
        
        // Normalize activations to [0,1] range
        double minVal, maxVal;
        cv::minMaxLoc(cnnActivations, &minVal, &maxVal);
        if (maxVal > 0) {
            cnnActivations = (cnnActivations - minVal) / (maxVal - minVal);
        }
        
        // Resize to match input dimensions
        cv::resize(cnnActivations, cnnActivations, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    }
    
    // Calculate score for each node based on multiple criteria
    for (size_t i = 0; i < graphNodes.size(); i++) {
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
    }
    
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
    
    // Optional: Add a smaller number of random activations for diversity
    const int numRandomSpikes = numInitialSpikes / 5; // 20% of the total
    for (int i = 0; i < numRandomSpikes; i++) {
        int randomIdx = rand() % graphNodes.size();
        graphNodes[randomIdx]->incrementPotential(0.6f); // Moderate activation
        graphNodes[randomIdx]->checkAndFire();
    }
    
    // Log statistics of initial activations
    if (!activatedPositions.empty()) {
        // Calculate average position
        cv::Point2f avgPos(0, 0);
        for (const auto &pos : activatedPositions) {
            avgPos.x += pos.x;
            avgPos.y += pos.y;
        }
        avgPos.x /= activatedPositions.size();
        avgPos.y /= activatedPositions.size();
        
        // Calculate spatial distribution
        float avgDist = 0;
        for (const auto &pos : activatedPositions) {
            float dx = pos.x - avgPos.x;
            float dy = pos.y - avgPos.y;
            avgDist += std::sqrt(dx*dx + dy*dy);
        }
        avgDist /= activatedPositions.size();
        
        std::cout << "NSGS: Initial activations avg position: (" << avgPos.x << ", " << avgPos.y 
                  << "), avg spread: " << avgDist << " pixels" << std::endl;
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
    // Implementation moved to reconstructFromNeuralGraph.cpp
    // This method is kept as a stub to maintain compatibility
    // See that file for the enhanced implementation using spatial clustering
    
    // Fallback implementation in case main implementation unavailable
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
    std::cout << "NSGS: Running asynchronous event processing with edge-based priority scheduling..." << std::endl;
    
    // Extract embeddings from the model output
    // In ONNX models, mask embeddings/protos are typically the second output tensor
    if (outputTensors.size() < 2 || !this->hasMask) {
        std::cout << "NSGS: No mask embeddings available, using basic processing" << std::endl;
        return;
    }
    
    // Get embeddings from the model's proto output (typically shape [1, proto_dim, H, W])
    float* maskOutput = outputTensors[1].GetTensorMutableData<float>();
    std::vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
    int protoChannels = maskShape[1]; // Number of proto/embedding channels
    int protoHeight = maskShape[2];
    int protoWidth = maskShape[3];
    
    std::cout << "NSGS: Using embeddings with shape [" << protoChannels << ", " 
              << protoHeight << ", " << protoWidth << "]" << std::endl;
    
    // Create proper embedding matrix from ONNX model's proto tensors
    cv::Mat embeddings(protoHeight, protoWidth, CV_32FC(protoChannels), maskOutput);
    
    // Initialize node potentials from embeddings - sets initial activity levels
    initializeNodePotentials(embeddings);
    
    // Process the main detection output to get high-confidence regions for seeding
    float* boxOutput = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
    float* output0ptr = (float *)output0.data;
    int rows = (int)this->outputShapes[0][2];
    int cols = (int)this->outputShapes[0][1];
    
    // Process detection boxes to seed initial class assignments
    std::vector<int> seedClassIds;
    std::vector<float> seedConfidences;
    std::vector<cv::Point2f> seedPositions;
    std::vector<cv::Rect> seedBoxes;
    
    for (int i = 0; i < rows; i++) {
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);
        
        if (confidence > this->confThreshold * 1.2f) { // Only use high-confidence detections
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            
            seedClassIds.push_back(classId);
            seedConfidences.push_back(confidence);
            seedPositions.push_back(cv::Point2f(centerX, centerY));
            seedBoxes.push_back(cv::Rect(left, top, width, height));
        }
    }
    
    // Seed nodes with class assignments based on detection boxes
    if (!seedPositions.empty()) {
        const int maxSeeds = 50; // Limit number of seed points for performance
        int numSeeds = std::min(maxSeeds, (int)seedPositions.size());
        
        for (int i = 0; i < numSeeds; i++) {
            // Find closest node to this seed position
            float minDist = std::numeric_limits<float>::max();
            int bestNodeIdx = -1;
            
            for (size_t j = 0; j < graphNodes.size(); j++) {
                cv::Point2i nodePos = graphNodes[j]->getPosition();
                float dist = cv::norm(cv::Point2f(nodePos.x, nodePos.y) - seedPositions[i]);
                
                if (dist < minDist) {
                    minDist = dist;
                    bestNodeIdx = j;
                }
            }
            
            // Assign class to the closest node
            if (bestNodeIdx >= 0) {
                graphNodes[bestNodeIdx]->setClassId(seedClassIds[i]);
                graphNodes[bestNodeIdx]->setConfidence(seedConfidences[i]);
                graphNodes[bestNodeIdx]->incrementPotential(1.0f);
                
                // Also find and assign to a group of nodes within the detection box
                // This creates stronger seed regions based on feature similarity
                cv::Rect box = seedBoxes[i];
                const std::vector<float> &seedFeatures = graphNodes[bestNodeIdx]->getFeatures();
                
                // Assign class to nodes within the box that have similar features
                for (size_t j = 0; j < graphNodes.size(); j++) {
                    if (j == bestNodeIdx) continue; // Skip seed node
                    
                    cv::Point2i nodePos = graphNodes[j]->getPosition();
                    
                    // Check if node is inside detection box (with small margin)
                    if (box.contains(nodePos)) {
                        // Calculate feature similarity with seed node
                        const std::vector<float> &nodeFeatures = graphNodes[j]->getFeatures();
                        float similarity = 0.0f;
                        
                        if (!seedFeatures.empty() && !nodeFeatures.empty()) {
                            size_t minSize = std::min(seedFeatures.size(), nodeFeatures.size());
                            float dotProduct = 0.0f;
                            float normSeed = 0.0f, normNode = 0.0f;
                            
                            for (size_t k = 0; k < minSize; k++) {
                                dotProduct += seedFeatures[k] * nodeFeatures[k];
                                normSeed += seedFeatures[k] * seedFeatures[k];
                                normNode += nodeFeatures[k] * nodeFeatures[k];
                            }
                            
                            if (normSeed > 0 && normNode > 0) {
                                similarity = dotProduct / (std::sqrt(normSeed) * std::sqrt(normNode));
                                similarity = std::max(0.0f, similarity);
                            }
                        }
                        
                        // Only assign class if features are similar enough (prevents leaking across edges)
                        if (similarity > 0.7f) {
                            graphNodes[j]->setClassId(seedClassIds[i]);
                            // Lower confidence for propagated nodes
                            graphNodes[j]->setConfidence(seedConfidences[i] * similarity);
                            // Higher initial potential for more similar nodes
                            graphNodes[j]->incrementPotential(0.7f * similarity);
                        }
                    }
                }
            }
        }
    } else {
        // Fallback if no high-confidence detections were found
        std::cout << "NSGS: No high-confidence detections, using feature-based clustering" << std::endl;
        
        // Apply feature-based clustering to find natural segments
        // First compute distance matrix between node features
        const int maxClusterNodes = std::min((int)graphNodes.size(), 500);
        std::vector<int> nodeIndices(graphNodes.size());
        for (size_t i = 0; i < nodeIndices.size(); i++) {
            nodeIndices[i] = i;
        }
        
        // Randomly sample nodes if there are too many (for performance)
        if (graphNodes.size() > maxClusterNodes) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(nodeIndices.begin(), nodeIndices.end(), g);
            nodeIndices.resize(maxClusterNodes);
        }
        
        // Group nodes with similar features using a simple clustering approach
        const float similarityThreshold = 0.85f;
        const int minClusterSize = 5;
        
        std::vector<int> nodeClusterIds(nodeIndices.size(), -1);
        int nextClusterId = 0;
        
        for (size_t i = 0; i < nodeIndices.size(); i++) {
            if (nodeClusterIds[i] >= 0) continue; // Skip assigned nodes
            
            // Start a new cluster
            std::vector<size_t> clusterIndices;
            clusterIndices.push_back(i);
            nodeClusterIds[i] = nextClusterId;
            
            // Find similar nodes
            for (size_t j = 0; j < nodeIndices.size(); j++) {
                if (i == j || nodeClusterIds[j] >= 0) continue;
                
                const std::vector<float> &featuresI = graphNodes[nodeIndices[i]]->getFeatures();
                const std::vector<float> &featuresJ = graphNodes[nodeIndices[j]]->getFeatures();
                
                float similarity = 0.0f;
                if (!featuresI.empty() && !featuresJ.empty()) {
                    size_t minSize = std::min(featuresI.size(), featuresJ.size());
                    float dotProduct = 0.0f;
                    float normI = 0.0f, normJ = 0.0f;
                    
                    for (size_t k = 0; k < minSize; k++) {
                        dotProduct += featuresI[k] * featuresJ[k];
                        normI += featuresI[k] * featuresI[k];
                        normJ += featuresJ[k] * featuresJ[k];
                    }
                    
                    if (normI > 0 && normJ > 0) {
                        similarity = dotProduct / (std::sqrt(normI) * std::sqrt(normJ));
                    }
                }
                
                if (similarity > similarityThreshold) {
                    clusterIndices.push_back(j);
                    nodeClusterIds[j] = nextClusterId;
                }
            }
            
            // Only keep sufficiently large clusters
            if (clusterIndices.size() >= minClusterSize) {
                // Assign a random class ID to this cluster
                int seedClass = nextClusterId % classNums;
                float seedConfidence = 0.7f;
                
                for (size_t idx : clusterIndices) {
                    int nodeIdx = nodeIndices[idx];
                    graphNodes[nodeIdx]->setClassId(seedClass);
                    graphNodes[nodeIdx]->setConfidence(seedConfidence);
                    graphNodes[nodeIdx]->incrementPotential(0.8f);
                }
                
                nextClusterId++;
            } else {
                // Reset small clusters
                for (size_t j = 0; j < nodeClusterIds.size(); j++) {
                    if (nodeClusterIds[j] == nextClusterId) {
                        nodeClusterIds[j] = -1;
                    }
                }
            }
        }
        
        std::cout << "NSGS: Created " << nextClusterId << " feature-based clusters" << std::endl;
    }
    
    // Start spike propagation to spread class assignments through the graph
    propagateSpikes(true);
    
    // Wait for spike processing to complete (with timeout)
    // This is now using priority-based scheduling which processes edge spikes first
    const int maxIterations = 20; // Max iterations to prevent infinite loops
    int iterations = 0;
    
    while (iterations < maxIterations && !spikeQueue->isEmpty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        iterations++;
    }
    
    std::cout << "NSGS: Processed " << spikeQueue->getProcessedCount() << " spikes in " 
              << iterations << " iterations using priority scheduling" << std::endl;
    std::cout << "NSGS: Max queue size was " << spikeQueue->getHighWatermark() << std::endl;
}

std::vector<Yolov8Result> NsgsPredictor::postprocessing(const cv::Size &resizedImageShape,
                                                      const cv::Size &originalImageShape,
                                                      std::vector<Ort::Value> &outputTensors)
{
    // Store the output tensors for use in NSGS processing
    this->outputTensors.clear();
    for (auto& tensor : outputTensors) {
        this->outputTensors.push_back(std::move(tensor));
    }
    
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
    if (this->hasMask && outputTensors.size() > 1)
    {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        std::vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int> mask_protos_shape = {1, (int)maskShape[1], (int)maskShape[2], (int)maskShape[3]};
        mask_protos = cv::Mat(mask_protos_shape, CV_32F, maskOutput);
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
    runAsyncEventProcessing();
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
                cv::Mat cnnMask = this->getMask(cv::Mat(picked_proposals[idx]).t(), mask_protos);
                
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
    
    // Original synchronous processing
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
        
        // Wait for processing to complete
        while (true) {
            bool allDone = true;
            
            // Check if any partition has active work
            for (auto& partition : partitions) {
                // Implement a check here - for now, just assume partitions are done
                // This should actually check the local SpikeQueue of each partition
            }
            
            if (allDone) break;
            
            // Synchronize boundary nodes
            syncPartitionBoundaries();
            
            // Sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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