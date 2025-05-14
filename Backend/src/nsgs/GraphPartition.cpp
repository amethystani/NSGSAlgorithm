#include "NsgsPredictor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

GraphPartition::GraphPartition(int id, std::vector<std::shared_ptr<NeuronNode>>* globalNodes)
    : partitionId(id),
      active(false),
      globalNodesRef(globalNodes)
{
    // Create a local spike queue for this partition
    int numThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 4);
    localSpikeQueue = std::make_shared<SpikeQueue>(numThreads);
    
    std::cout << "NSGS: Created graph partition " << id 
              << " with " << numThreads << " processing threads" << std::endl;
}

GraphPartition::~GraphPartition()
{
    stop();
}

void GraphPartition::addNode(std::shared_ptr<NeuronNode> node, bool isBoundary)
{
    // Add node to this partition
    nodes.push_back(node);
    
    // If this is a boundary node, record its index
    if (isBoundary) {
        boundaryNodeIndices.push_back(nodes.size() - 1);
    }
}

void GraphPartition::addExternalConnection(int localNodeIndex, int remoteNodeGlobalIndex)
{
    if (localNodeIndex >= 0 && localNodeIndex < static_cast<int>(nodes.size())) {
        externalConnections.push_back(std::make_pair(localNodeIndex, remoteNodeGlobalIndex));
    }
}

void GraphPartition::syncBoundaryNodes()
{
    std::lock_guard<std::mutex> lock(boundaryMutex);
    
    // Skip if there are no global nodes reference
    if (!globalNodesRef) return;
    
    // For each boundary node, synchronize with its external connections
    for (int localIdx : boundaryNodeIndices) {
        if (localIdx < 0 || localIdx >= static_cast<int>(nodes.size())) continue;
        
        auto& node = nodes[localIdx];
        int classId = node->getClassId();
        float confidence = node->getConfidence();
        
        // Only propagate high-confidence classifications across partition boundaries
        if (classId >= 0 && confidence > 0.7f) {
            // Find all external connections for this node
            for (const auto& conn : externalConnections) {
                if (conn.first == localIdx) {
                    int remoteIdx = conn.second;
                    
                    // Check if the remote index is valid
                    if (remoteIdx >= 0 && remoteIdx < static_cast<int>(globalNodesRef->size())) {
                        auto& remoteNode = (*globalNodesRef)[remoteIdx];
                        
                        // If remote node has no class or lower confidence, propagate our class
                        int remoteClassId = remoteNode->getClassId();
                        float remoteConfidence = remoteNode->getConfidence();
                        
                        if (remoteClassId < 0 || remoteConfidence < confidence) {
                            // Calculate feature similarity to determine if we should propagate
                            float similarity = 0.0f;
                            
                            const std::vector<float>& nodeFeatures = node->getFeatures();
                            const std::vector<float>& remoteFeatures = remoteNode->getFeatures();
                            
                            if (!nodeFeatures.empty() && !remoteFeatures.empty()) {
                                // Calculate cosine similarity
                                size_t minSize = std::min(nodeFeatures.size(), remoteFeatures.size());
                                float dotProduct = 0.0f;
                                float normA = 0.0f;
                                float normB = 0.0f;
                                
                                for (size_t i = 0; i < minSize; i++) {
                                    dotProduct += nodeFeatures[i] * remoteFeatures[i];
                                    normA += nodeFeatures[i] * nodeFeatures[i];
                                    normB += remoteFeatures[i] * remoteFeatures[i];
                                }
                                
                                normA = std::sqrt(normA);
                                normB = std::sqrt(normB);
                                
                                if (normA > 0.0f && normB > 0.0f) {
                                    similarity = dotProduct / (normA * normB);
                                }
                            }
                            
                            // Only propagate to similar nodes
                            if (similarity > 0.7f) {
                                // Directly set the class ID and confidence
                                remoteNode->setClassId(classId);
                                remoteNode->setConfidence(confidence * 0.9f);
                            }
                        }
                    }
                }
            }
        }
    }
}

void GraphPartition::start()
{
    if (active.load()) return;
    
    active.store(true);
    
    // Start the local spike queue
    localSpikeQueue->startProcessing(&nodes);
    
    // Start the processing thread
    processingThread = std::thread(&GraphPartition::processPartition, this);
    
    std::cout << "NSGS: Started partition " << partitionId 
              << " with " << nodes.size() << " nodes ("
              << boundaryNodeIndices.size() << " boundary nodes)" << std::endl;
}

void GraphPartition::stop()
{
    active.store(false);
    
    // Stop the local spike queue
    localSpikeQueue->stopProcessing();
    
    // Wait for processing thread to finish
    if (processingThread.joinable()) {
        processingThread.join();
    }
    
    std::cout << "NSGS: Stopped partition " << partitionId << std::endl;
}

void GraphPartition::processPartition()
{
    // Initial seeding for this partition - find high-contrast nodes
    std::vector<std::pair<int, float>> nodeScores;
    
    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];
        const std::vector<float>& features = node->getFeatures();
        
        float score = 0.0f;
        
        // Use gradient features for contrast detection
        if (features.size() > 19) {
            // Assuming gradient features are at indices 19-26
            for (size_t j = 19; j < std::min(size_t(27), features.size()); j++) {
                score += features[j];
            }
            score /= 8.0f; // Normalize
        }
        
        nodeScores.push_back(std::make_pair(i, score));
    }
    
    // Sort nodes by score
    std::sort(nodeScores.begin(), nodeScores.end(), 
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });
    
    // Seed the top N nodes
    int numInitialSpikes = std::max(5, static_cast<int>(nodes.size() / 50));
    for (int i = 0; i < std::min(numInitialSpikes, static_cast<int>(nodeScores.size())); i++) {
        int nodeIdx = nodeScores[i].first;
        nodes[nodeIdx]->incrementPotential(0.7f);
        nodes[nodeIdx]->checkAndFire();
    }
    
    // Process until stopped
    auto lastBoundarySyncTime = std::chrono::steady_clock::now();
    
    while (active.load()) {
        // Sleep briefly to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Periodically synchronize boundary nodes
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastBoundarySyncTime).count();
        
        if (elapsed > 100) { // Sync every 100ms
            syncBoundaryNodes();
            lastBoundarySyncTime = now;
        }
    }
}