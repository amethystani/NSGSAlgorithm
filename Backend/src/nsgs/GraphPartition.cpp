#include "NsgsPredictor.h"
#include <iostream>
#include <thread>
#include <stdexcept>

GraphPartition::GraphPartition(int id, std::vector<std::shared_ptr<NeuronNode>>* globalNodes)
    : partitionId(id), globalNodesRef(globalNodes), active(false)
{
    // Create local spike queue
    localSpikeQueue = std::make_shared<SpikeQueue>();
    
    std::cout << "Created graph partition " << id << std::endl;
}

GraphPartition::~GraphPartition()
{
    // Make sure processing is stopped before destruction
    stop();
    
    std::cout << "Destroyed graph partition " << partitionId << std::endl;
}

void GraphPartition::addNode(std::shared_ptr<NeuronNode> node, bool isBoundary)
{
    // Safety check for null node pointer
    if (!node) {
        std::cerr << "GraphPartition: Cannot add null node" << std::endl;
        return;
    }
    
    // Add the node to this partition
    nodes.push_back(node);
    
    // Mark as boundary node if specified
    if (isBoundary) {
        boundaryNodeIndices.push_back(nodes.size() - 1);
    }
}

void GraphPartition::addExternalConnection(int localNodeIndex, int remoteNodeGlobalIndex)
{
    // Safety checks for valid indices
    if (localNodeIndex < 0 || localNodeIndex >= static_cast<int>(nodes.size())) {
        std::cerr << "GraphPartition: Invalid local node index: " << localNodeIndex << std::endl;
        return;
    }
    
    if (!globalNodesRef) {
        std::cerr << "GraphPartition: No global nodes reference available" << std::endl;
        return;
    }
    
    if (remoteNodeGlobalIndex < 0 || remoteNodeGlobalIndex >= static_cast<int>(globalNodesRef->size())) {
        std::cerr << "GraphPartition: Invalid remote node index: " << remoteNodeGlobalIndex << std::endl;
        return;
    }
    
    // Add external connection
    externalConnections.emplace_back(localNodeIndex, remoteNodeGlobalIndex);
}

void GraphPartition::syncBoundaryNodes()
{
    // Thread-safe protection for boundary nodes
    std::lock_guard<std::mutex> lock(boundaryMutex);
    
    if (!globalNodesRef) {
        std::cerr << "GraphPartition: No global nodes reference available for sync" << std::endl;
        return;
    }
    
    // Process each boundary node
    for (int index : boundaryNodeIndices) {
        // Safety check
        if (index < 0 || index >= static_cast<int>(nodes.size())) {
            continue;
        }
        
        auto& node = nodes[index];
        if (!node) continue;
        
        // Get current node state
        int classId = node->getClassId();
        float confidence = node->getConfidence();
        float potential = node->getPotential();
        
        // Use ID from node to find the global node
        int nodeId = node->getId();
        if (nodeId >= 0 && nodeId < static_cast<int>(globalNodesRef->size()) && (*globalNodesRef)[nodeId]) {
            // Copy state to global node if it has higher confidence
            auto& globalNode = (*globalNodesRef)[nodeId];
            
            // Only propagate state if this has valid classification with good confidence
            if (classId >= 0 && confidence > 0.4f) {
                if (classId != globalNode->getClassId() || confidence > globalNode->getConfidence()) {
                    globalNode->setClassId(classId);
                    globalNode->setConfidence(confidence);
                    
                    // Also share potential (take max of current and new)
                    float globalPotential = globalNode->getPotential();
                    if (potential > globalPotential) {
                        globalNode->incrementPotential(potential - globalPotential);
                    }
                }
            }
            
            // Copy from global to local if needed
            if (globalNode->getClassId() >= 0 && 
                (classId < 0 || globalNode->getConfidence() > node->getConfidence())) {
                node->setClassId(globalNode->getClassId());
                node->setConfidence(globalNode->getConfidence());
                
                // Also share potential (take max of current and new)
                float globalPotential = globalNode->getPotential();
                if (globalPotential > potential) {
                    node->incrementPotential(globalPotential - potential);
                }
            }
        }
    }
}

void GraphPartition::start()
{
    // Don't start if already active
    if (active.load()) {
        return;
    }
    
    // Activate partition
    active.store(true);
    
    // Start processing thread
    processingThread = std::thread(&GraphPartition::processPartition, this);
}

void GraphPartition::stop() {
    // Mark partition as inactive
    active.store(false);
    
    // Wait for thread to complete
    if (processingThread.joinable()) {
        processingThread.join();
    }
}

void GraphPartition::processPartition()
{
    // Safety check
    if (!localSpikeQueue || nodes.empty() || !globalNodesRef) {
        active.store(false);
        return;
    }
    
    try {
        localSpikeQueue->startProcessing(&nodes);
        
        // Initial activation - seed random nodes to start activity
        int numInitialSpikes = std::min(5, static_cast<int>(nodes.size() / 10));
        if (numInitialSpikes > 0 && !nodes.empty()) {
            for (int i = 0; i < numInitialSpikes; i++) {
                int randomIndex = rand() % nodes.size();
                nodes[randomIndex]->incrementPotential(0.6f);
                nodes[randomIndex]->checkAndFire();
            }
        }
        
        // Main processing loop
        int iteration = 0;
        const int maxIterations = 100;  // Limit to prevent infinite loops
        
        while (active.load() && iteration < maxIterations) {
            // Process each node
            for (auto& node : nodes) {
                if (!node) continue;
                
                // Check if node should fire based on current potential
                node->checkAndFire();
            }
            
            // Process external connections
            for (const auto& conn : externalConnections) {
                // Safety check for valid indices
                if (conn.first >= 0 && conn.first < static_cast<int>(nodes.size()) &&
                    conn.second >= 0 && conn.second < static_cast<int>(globalNodesRef->size())) {
                    
                    auto& localNode = nodes[conn.first];
                    auto& remoteNode = (*globalNodesRef)[conn.second];
                    
                    // Skip invalid nodes
                    if (!localNode || !remoteNode) continue;
                    
                    // If local node has class ID, propagate to remote node
                    int localClassId = localNode->getClassId();
                    if (localClassId >= 0) {
                        float confidence = localNode->getConfidence();
                        remoteNode->receiveSpike(0.3f, localClassId, confidence * 0.8f);
                    }
                }
            }
            
            // Synchronize with other partitions periodically
            if (iteration % 10 == 0) {
                syncBoundaryNodes();
            }
            
            // Sleep briefly to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            iteration++;
        }
        
        // Final sync before stopping
        syncBoundaryNodes();
        
        // Stop local spike queue
        localSpikeQueue->stopProcessing();
        
    } catch (const std::exception& e) {
        std::cerr << "GraphPartition " << partitionId << " exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "GraphPartition " << partitionId << " unknown exception" << std::endl;
    }
}