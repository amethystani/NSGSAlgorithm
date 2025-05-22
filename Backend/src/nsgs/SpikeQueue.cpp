#include "SpikeQueue.h"
#include "NeuronNode.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <random>
#include <cmath>

// Initialize thread_local ID
thread_local int SpikeQueue::threadLocalId = -1;

SpikeQueue::SpikeQueue(int numThreads)
    : processing(false),
      terminated(false),
      nodesRef(nullptr),
      processedSpikeCount(0),
      queueHighWatermark(0),
      totalQueuedSpikes(0),
      lastAdaptationTime(std::chrono::steady_clock::now()),
      stateChangeCount(0),
      hasRecentActivity(false),
      hasConverged(false),
      lastActivityTime(std::chrono::steady_clock::now()),
      numThreads(numThreads),
      activeWorkers(0),
      nextThreadId(0)
{
    // Validate thread count and adjust if needed
    this->numThreads = std::max(1, numThreads);
    
    // Hardware concurrency check - limit to available cores if too high
    unsigned int hwThreads = std::thread::hardware_concurrency();
    if (hwThreads > 0 && this->numThreads > static_cast<int>(hwThreads)) {
        this->numThreads = hwThreads;
    }
    
    // Initialize thread queues
    threadQueues.resize(this->numThreads);
    for (int i = 0; i < this->numThreads; i++) {
        threadQueues[i] = std::make_unique<WorkStealingDeque>();
    }
    
    std::cout << "NSGS: Initializing SpikeQueue with " << this->numThreads 
              << " worker threads and lock-free work-stealing" << std::endl;
}

SpikeQueue::~SpikeQueue()
{
    // Make sure the processing threads are stopped
    stopProcessing();
}

void SpikeQueue::addSpike(const Spike& spike)
{
    // Add the spike to the global queue
    {
        std::lock_guard<std::mutex> lock(globalQueueMutex);
        globalQueue.push(spike);
        
        // Update total queued counter
        totalQueuedSpikes.fetch_add(1);
        
        // Update high watermark (global + all thread queues)
        size_t totalSize = globalQueue.size();
        for (const auto& queue : threadQueues) {
            totalSize += queue->size();
        }
        
        size_t currentHighWatermark = queueHighWatermark.load();
        if (totalSize > currentHighWatermark) {
            queueHighWatermark.store(totalSize);
        }
    }
    
    // Record activity timestamp - a spike was added
    lastActivityTime = std::chrono::steady_clock::now();
    hasRecentActivity.store(true);
    
    // Notify waiting threads
    globalQueueCondition.notify_one();
}

void SpikeQueue::addSpike(int sourceNodeId, int targetNodeId, float weight)
{
    // Calculate priority for this spike
    float priority = calculateSpikePriority(sourceNodeId, targetNodeId, weight);
    
    // Create and add the spike with priority
    Spike spike(sourceNodeId, targetNodeId, weight, priority);
    addSpike(spike);
}

Spike SpikeQueue::getNextSpike()
{
    std::unique_lock<std::mutex> lock(globalQueueMutex);
    
    // Wait for queue to have an item
    globalQueueCondition.wait(lock, [this]() {
        return !globalQueue.empty() || terminated.load();
    });
    
    if (terminated.load()) {
        return Spike(); // Return empty spike if terminated
    }
    
    // Get the spike from the front of the queue
    Spike spike = globalQueue.top();
    globalQueue.pop();
    
    totalQueuedSpikes.fetch_sub(1);
    
    return spike;
}

bool SpikeQueue::tryGetNextSpike(Spike& spike)
{
    std::unique_lock<std::mutex> lock(globalQueueMutex);
    
    if (globalQueue.empty() || terminated.load()) {
        return false;
    }
    
    spike = globalQueue.top();
    globalQueue.pop();
    
    totalQueuedSpikes.fetch_sub(1);
    
    return true;
}

bool SpikeQueue::getLocalWork(int threadId, Spike& spike)
{
    // Try to get work from this thread's own queue
    if (threadId >= 0 && threadId < static_cast<int>(threadQueues.size())) {
        if (threadQueues[threadId]->pop(spike)) {
            return true;
        }
    }
    return false;
}

bool SpikeQueue::stealWork(int threadId, Spike& spike)
{
    // Try to steal work from other threads' queues
    
    if (threadQueues.empty()) return false;
    
    // Use a random victim selection strategy to reduce contention
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, threadQueues.size() - 1);
    
    // Try to steal from 3 random victims before giving up
    for (int attempts = 0; attempts < 3; attempts++) {
        int victimId = dis(gen);
        
        // Don't try to steal from our own queue
        if (victimId == threadId) {
            continue;
        }
        
        if (threadQueues[victimId]->steal(spike)) {
            return true;
        }
    }
    
    return false;
}

bool SpikeQueue::getGlobalWork(Spike& spike)
{
    std::unique_lock<std::mutex> lock(globalQueueMutex);
    
    // If global queue has items, distribute some to local queues for better parallelism
    if (!globalQueue.empty()) {
        // If there are more than 2 spikes per thread, distribute work
        const size_t threshold = numThreads * 2;
        
        if (globalQueue.size() > threshold) {
            for (size_t i = 0; i < numThreads; i++) {
                if (!globalQueue.empty()) {
                    spike = globalQueue.top();
                    globalQueue.pop();
                    
                    // First, use one spike for this thread
                    if (i == 0) {
                        lock.unlock();
                        return true;
                    }
                    
                    // Distribute others to thread-local queues
                    threadQueues[i]->push(globalQueue.top());
                    globalQueue.pop();
                    totalQueuedSpikes.fetch_sub(1);
                }
            }
        }
        
        // Get one spike for this thread
        if (!globalQueue.empty()) {
            spike = globalQueue.top();
            globalQueue.pop();
            totalQueuedSpikes.fetch_sub(1);
            return true;
        }
    }
    
    return false;
}

void SpikeQueue::balanceLoad()
{
    // Check if load is imbalanced
    size_t maxQueueSize = 0;
    size_t minQueueSize = std::numeric_limits<size_t>::max();
    size_t totalSize = 0;
    
    for (const auto& queue : threadQueues) {
        size_t size = queue->size();
        maxQueueSize = std::max(maxQueueSize, size);
        minQueueSize = std::min(minQueueSize, size);
        totalSize += size;
    }
    
    // Skip if load is balanced
    if (totalSize == 0 || maxQueueSize < 3 * minQueueSize) {
        return;
    }
    
    // Log imbalance
    std::cout << "NSGS: Load imbalance detected - min queue size: " << minQueueSize
              << ", max queue size: " << maxQueueSize << std::endl;
    
    // Balance is handled automatically by work stealing in the worker threads
}

void SpikeQueue::startProcessing(std::vector<std::shared_ptr<NeuronNode>>* nodes)
{
    // Safety check for nullptr
    if (!nodes) {
        std::cerr << "SpikeQueue: Null node vector pointer provided" << std::endl;
        return;
    }
    
    // Store reference to nodes
    nodesRef = nodes;
    
    // If already processing, do nothing
    if (processing.load()) {
        return;
    }
    
    // Reset termination flags
    terminated.store(false);
    hasConverged.store(false);
    
    // Set processing flag
    processing.store(true);
    
    // Reset counters
    processedSpikeCount.store(0);
    totalQueuedSpikes.store(0);
    queueHighWatermark.store(0);
    stateChangeCount.store(0);
    activeWorkers.store(0);
    nextThreadId.store(0);
    
    // Clear convergence history
    {
        std::lock_guard<std::mutex> lock(convergenceHistoryMutex);
        convergenceHistory.clear();
    }
    
    // Initialize activity tracking
    hasRecentActivity.store(true);
    lastActivityTime = std::chrono::steady_clock::now();
    
    // Clear all thread queues
    for (auto& queue : threadQueues) {
        Spike dummy;
        while (queue->pop(dummy)) {
            // Just drain the queue
        }
    }
    
    // Start the worker threads
    workerThreads.clear();
    for (int i = 0; i < numThreads; i++) {
        workerThreads.push_back(std::thread(&SpikeQueue::workerThreadFunction, this, i));
    }
    
    std::cout << "NSGS: Started " << numThreads << " worker threads with lock-free work-stealing" << std::endl;
}

void SpikeQueue::stopProcessing()
{
    // Set termination flag
    terminated.store(true);
    
    // Wake up any waiting threads
    globalQueueCondition.notify_all();
    terminationCondition.notify_all();
    
    // Wait for all threads to finish
    for (auto& thread : workerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Clear the queue
    clear();
    
    // Reset processing flag
    processing.store(false);
    
    // Clear worker threads
    workerThreads.clear();
}

void SpikeQueue::clear()
{
    std::unique_lock<std::mutex> lock(globalQueueMutex);
    
    // Clear global queue
    std::priority_queue<Spike> empty;
    globalQueue = std::priority_queue<Spike>(); // Create a new empty queue
    
    totalQueuedSpikes.store(0);
    processedSpikeCount.store(0);
    queueHighWatermark.store(0);
    stateChangeCount.store(0);
    
    // Clear convergence history
    {
        std::lock_guard<std::mutex> historyLock(convergenceHistoryMutex);
        convergenceHistory.clear();
    }
    
    // Reset activity tracking
    hasRecentActivity.store(false);
    hasConverged.store(false);
    lastActivityTime = std::chrono::steady_clock::now();
}

bool SpikeQueue::isEmpty() const
{
    // Check global queue
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(globalQueueMutex));
        if (!globalQueue.empty()) {
            return false;
        }
    }
    
    // Check all thread queues
    for (const auto& queue : threadQueues) {
        if (!queue->isEmpty()) {
            return false;
        }
    }
    
    return true;
}

size_t SpikeQueue::size() const
{
    size_t total = 0;
    
    // Count spikes in global queue
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(globalQueueMutex));
        total += globalQueue.size();
    }
    
    // Count spikes in all thread queues
    for (const auto& queue : threadQueues) {
        total += queue->size();
    }
    
    return total;
}

void SpikeQueue::setThermalFeedbackCallback(std::function<void(float)> callback)
{
    thermalFeedbackCallback = callback;
}

void SpikeQueue::checkThermalAdaptation()
{
    // Check if it's time to adapt
    auto now = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAdaptationTime).count();
    
    // Adapt every 100ms
    if (elapsedMs > 100) {
        lastAdaptationTime = now;
        
        // Calculate load based on queue size and processed count
        size_t currentSize = size();
        size_t processedCount = processedSpikeCount.load();
        
        // Simple load metric: queue size relative to processed count
        float load = 0.0f;
        if (processedCount > 0) {
            load = std::min(1.0f, static_cast<float>(currentSize) / static_cast<float>(processedCount));
        }
        
        // Call the callback if available
        if (thermalFeedbackCallback) {
            thermalFeedbackCallback(load);
        }
        
        // Also record convergence data
        recordConvergenceData();
        
        // Check load balance
        balanceLoad();
        
        // Check for convergence
        if (checkConvergence()) {
            std::cout << "NSGS: Convergence detected, preparing for termination" << std::endl;
            hasConverged.store(true);
            globalQueueCondition.notify_all(); // Wake up any waiting threads
            terminationCondition.notify_all(); // Wake up worker threads
        }
    }
}

void SpikeQueue::recordStateChange()
{
    // Increment the state change counter
    stateChangeCount.fetch_add(1);
    
    // Record activity
    lastActivityTime = std::chrono::steady_clock::now();
    hasRecentActivity.store(true);
}

float SpikeQueue::getConvergenceRate() const
{
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(convergenceHistoryMutex));
    
    if (convergenceHistory.size() < 2) {
        return 1.0f; // Not enough data to calculate rate
    }
    
    // Calculate rate of change in activity
    const auto& newest = convergenceHistory.front();
    const auto& oldest = convergenceHistory.back();
    
    // Time difference in seconds
    float timeDiffSec = std::chrono::duration_cast<std::chrono::milliseconds>(
        newest.timestamp - oldest.timestamp).count() / 1000.0f;
    
    if (timeDiffSec < 0.001f) return 1.0f; // Avoid division by zero
    
    // Changes per second
    float processedDiff = static_cast<float>(newest.processedSpikes - oldest.processedSpikes) / timeDiffSec;
    float stateChangeDiff = static_cast<float>(newest.stateChanges - oldest.stateChanges) / timeDiffSec;
    
    // Combined rate (normalized)
    float totalInitialActivity = std::max(1.0f, static_cast<float>(processedSpikeCount.load()) / 10.0f);
    float normalizedRate = (processedDiff + stateChangeDiff * 5.0f) / totalInitialActivity;
    
    return std::min(1.0f, std::max(0.0f, normalizedRate));
}

std::chrono::milliseconds SpikeQueue::getTimeSinceLastActivity() const
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActivityTime);
}

bool SpikeQueue::waitForConvergence(int maxWaitTimeMs)
{
    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::milliseconds(maxWaitTimeMs);
    
    // Check periodically until timeout or convergence
    while (std::chrono::steady_clock::now() < endTime) {
        if (hasConverged.load() || terminated.load()) {
            return hasConverged.load();
        }
        
        // Also check inactivity-based convergence
        auto inactivityTime = getTimeSinceLastActivity();
        if (inactivityTime.count() > inactivityThresholdMs && isEmpty()) {
            hasConverged.store(true);
            std::cout << "NSGS: No activity for " << inactivityTime.count() 
                      << "ms, system considered converged" << std::endl;
            return true;
        }
        
        // Sleep briefly to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Timeout occurred
    std::cout << "NSGS: Convergence wait timed out after " << maxWaitTimeMs << "ms" << std::endl;
    return false;
}

void SpikeQueue::recordConvergenceData()
{
    std::lock_guard<std::mutex> lock(convergenceHistoryMutex);
    
    // Create new data point
    ConvergenceData data(
        size(),
        processedSpikeCount.load(),
        stateChangeCount.load()
    );
    
    // Add to history (front = newest)
    convergenceHistory.push_front(data);
    
    // Limit history size
    if (convergenceHistory.size() > convergenceHistorySize) {
        convergenceHistory.pop_back();
    }
}

bool SpikeQueue::checkConvergence()
{
    // Check if we've processed enough spikes to even consider convergence
    if (processedSpikeCount.load() < 100) {
        return false;
    }
    
    // We need a minimum amount of history
    {
        std::lock_guard<std::mutex> lock(convergenceHistoryMutex);
        if (convergenceHistory.size() < convergenceHistorySize / 2) {
            return false;
        }
    }
    
    // Method 1: Check convergence rate (activity slowed down)
    float rate = getConvergenceRate();
    if (rate < activityThreshold) {
        std::cout << "NSGS: Activity rate " << rate << " below threshold " 
                  << activityThreshold << ", system considered converged" << std::endl;
        return true;
    }
    
    // Method 2: Check inactivity period (no new spikes or state changes)
    auto inactivityTime = getTimeSinceLastActivity();
    if (inactivityTime.count() > inactivityThresholdMs && isEmpty()) {
        std::cout << "NSGS: No activity for " << inactivityTime.count() 
                  << "ms, system considered converged" << std::endl;
        return true;
    }
    
    // Method 3: Queue is empty and we've processed a significant number of spikes
    if (isEmpty() && processedSpikeCount.load() > 1000) {
        std::cout << "NSGS: Queue empty after processing " << processedSpikeCount.load() 
                  << " spikes, system considered converged" << std::endl;
        return true;
    }
    
    return false;
}

void SpikeQueue::updateActivity()
{
    // Reset activity flag if too much time has passed since last activity
    auto now = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActivityTime).count();
    
    if (elapsedMs > inactivityThresholdMs) {
        hasRecentActivity.store(false);
    }
}

// Process a single spike
void SpikeQueue::processSingleSpike(const Spike& spike)
{
    // Process the spike
    if (nodesRef && spike.targetNodeId >= 0 && static_cast<size_t>(spike.targetNodeId) < nodesRef->size()) {
        // Get the target node
        auto& targetNode = (*nodesRef)[spike.targetNodeId];
        if (!targetNode) return;

        // Get the source node for class information
        int sourceClassId = -1;
        float sourceConfidence = 0.0f;
        if (spike.sourceNodeId >= 0 && static_cast<size_t>(spike.sourceNodeId) < nodesRef->size()) {
            auto& sourceNode = (*nodesRef)[spike.sourceNodeId];
            if (sourceNode) {
                sourceClassId = sourceNode->getClassId();
                sourceConfidence = sourceNode->getConfidence();
            }
        }

        // Record node's current state (for convergence detection)
        int oldClassId = targetNode->getClassId();
        float oldConfidence = targetNode->getConfidence();
        
        // Deliver the spike with class information
        targetNode->receiveSpike(spike.weight, sourceClassId, sourceConfidence);
        
        // Check if the node changed state
        if (oldClassId != targetNode->getClassId() || 
            (oldClassId >= 0 && std::abs(oldConfidence - targetNode->getConfidence()) > 0.1f)) {
            // State changed significantly, record it
            recordStateChange();
        }
        
        // Increment the processed count
        processedSpikeCount.fetch_add(1);
        
        // Update activity timestamp
        lastActivityTime = std::chrono::steady_clock::now();
        hasRecentActivity.store(true);
    }
}

// Worker thread function - each thread in the pool runs this
void SpikeQueue::workerThreadFunction(int threadId)
{
    // Set thread-local ID for work stealing
    threadLocalId = threadId;
    
    // Record that this worker is active
    activeWorkers.fetch_add(1);
    
    std::cout << "NSGS: Worker thread " << threadId << " started" << std::endl;
    
    // Work processing loop
    while (!terminated.load() && !hasConverged.load()) {
        Spike spike(-1, -1, 0.0f);
        bool hasWork = false;
        
        // Work stealing approach: try local queue first, then steal, then global queue
        
        // 1. Try to get work from local queue
        if (threadId >= 0 && threadId < static_cast<int>(threadQueues.size())) {
            hasWork = threadQueues[threadId]->pop(spike);
        }
        
        // 2. If no local work, try to steal from other workers
        if (!hasWork) {
            hasWork = stealWork(threadId, spike);
        }
        
        // 3. If no work was stolen, try the global queue
        if (!hasWork) {
            hasWork = getGlobalWork(spike);
        }
        
        if (hasWork) {
            // Process the spike
            processSingleSpike(spike);
        } else {
            // No work available
            
            // Check if we have any work across all queues
            if (isEmpty()) {
                // Update activity state and check for inactivity-based convergence
                updateActivity();
                
                if (!hasRecentActivity.load()) {
                    // Check for convergence explicitly
                    if (checkConvergence()) {
                        hasConverged.store(true);
                        terminationCondition.notify_all();
                        break;
                    }
                }
                
                // Wait for more work or termination with a short timeout
                std::unique_lock<std::mutex> lock(terminationMutex);
                terminationCondition.wait_for(lock, std::chrono::milliseconds(10), [this]{
                    return !isEmpty() || terminated.load() || hasConverged.load();
                });
            } else {
                // There's work somewhere, just yield to let other threads run
                std::this_thread::yield();
            }
        }
        
        // Check thermal adaptation and convergence (only thread 0 does this)
        if (threadId == 0) {
            checkThermalAdaptation();
        }
    }
    
    // Worker is exiting
    activeWorkers.fetch_sub(1);
    
    std::cout << "NSGS: Worker thread " << threadId << " terminated" << std::endl;
}

void SpikeQueue::registerNodeEdgeStrength(int nodeId, float edgeStrength) {
    if (nodeId < 0) return;  // Skip invalid nodes
    
    std::lock_guard<std::mutex> lock(edgeStrengthMutex);
    
    // Register edge strength for node (used for all its connections)
    nodeEdgeStrengths[nodeId] = std::min(1.0f, std::max(0.0f, edgeStrength));
    
    // Update existing edges involving this node
    for (auto& edgePair : edgeStrengths) {
        if (edgePair.first.first == nodeId || edgePair.first.second == nodeId) {
            // Calculate average of node edge strengths
            int otherId = (edgePair.first.first == nodeId) ? edgePair.first.second : edgePair.first.first;
            float otherStrength = 0.0f;
            
            auto otherIt = nodeEdgeStrengths.find(otherId);
            if (otherIt != nodeEdgeStrengths.end()) {
                otherStrength = otherIt->second;
            }
            
            // Update edge strength (average of node strengths)
            edgePair.second = (nodeEdgeStrengths[nodeId] + otherStrength) / 2.0f;
        }
    }
}

void SpikeQueue::updateNodeEdgeStrength(int nodeId, float edgeStrength) {
    std::lock_guard<std::mutex> lock(edgeStrengthMutex);
    auto it = nodeEdgeStrengths.find(nodeId);
    if (it != nodeEdgeStrengths.end()) {
        it->second = std::max(it->second, edgeStrength); // Keep highest edge strength
    } else {
        nodeEdgeStrengths[nodeId] = edgeStrength;
    }
}

float SpikeQueue::calculateSpikePriority(int sourceNodeId, int targetNodeId, float weight) {
    float priority = weight; // Base priority on weight
    
    // Add edge strength priority if available
    {
        std::lock_guard<std::mutex> lock(edgeStrengthMutex);
        
        auto sourceIt = nodeEdgeStrengths.find(sourceNodeId);
        if (sourceIt != nodeEdgeStrengths.end()) {
            // Higher edge strength = higher priority
            priority += sourceIt->second * 2.0f; // Emphasize edge strength from source
        }
        
        auto targetIt = nodeEdgeStrengths.find(targetNodeId);
        if (targetIt != nodeEdgeStrengths.end()) {
            priority += targetIt->second * 1.0f; // Consider target edge strength as well
        }
    }
    
    return priority;
} 