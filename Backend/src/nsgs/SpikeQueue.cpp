#include "SpikeQueue.h"
#include "NeuronNode.h"
#include <iostream>

SpikeQueue::SpikeQueue()
    : processing(false),
      terminated(false),
      nodesRef(nullptr),
      processedSpikeCount(0),
      queueHighWatermark(0),
      lastAdaptationTime(std::chrono::steady_clock::now())
{
    // Initialize with empty queue
}

SpikeQueue::~SpikeQueue()
{
    // Make sure the processing thread is stopped
    stopProcessing();
}

void SpikeQueue::addSpike(const Spike& spike)
{
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        spikes.push(spike);
        
        // Update high watermark
        size_t currentSize = spikes.size();
        size_t currentHighWatermark = queueHighWatermark.load();
        if (currentSize > currentHighWatermark) {
            queueHighWatermark.store(currentSize);
        }
    }
    
    // Notify one waiting thread
    condition.notify_one();
}

void SpikeQueue::addSpike(int sourceNodeId, int targetNodeId, float weight)
{
    Spike spike(sourceNodeId, targetNodeId, weight);
    addSpike(spike);
}

Spike SpikeQueue::getNextSpike()
{
    std::unique_lock<std::mutex> lock(queueMutex);
    
    // Wait until there's a spike or we're terminated
    condition.wait(lock, [this]{ 
        return !spikes.empty() || terminated.load(); 
    });
    
    // If terminated, return a dummy spike
    if (terminated.load()) {
        return Spike(-1, -1, 0.0f);
    }
    
    // Get the next spike
    Spike spike = spikes.front();
    spikes.pop();
    
    return spike;
}

bool SpikeQueue::tryGetNextSpike(Spike& spike)
{
    std::lock_guard<std::mutex> lock(queueMutex);
    
    if (spikes.empty() || terminated.load()) {
        return false;
    }
    
    spike = spikes.front();
    spikes.pop();
    return true;
}

void SpikeQueue::startProcessing(std::vector<std::shared_ptr<NeuronNode>>* nodes)
{
    // Store reference to nodes
    nodesRef = nodes;
    
    // If already processing, do nothing
    if (processing.load()) {
        return;
    }
    
    // Reset termination flag
    terminated.store(false);
    
    // Set processing flag
    processing.store(true);
    
    // Reset counters
    processedSpikeCount.store(0);
    queueHighWatermark.store(0);
    
    // Start the processing thread
    processingThread = std::thread(&SpikeQueue::processingLoop, this);
}

void SpikeQueue::stopProcessing()
{
    // Set termination flag
    terminated.store(true);
    
    // Wake up any waiting threads
    condition.notify_all();
    
    // Wait for processing thread to finish
    if (processingThread.joinable()) {
        processingThread.join();
    }
    
    // Clear the queue
    clear();
    
    // Reset processing flag
    processing.store(false);
}

void SpikeQueue::clear()
{
    std::lock_guard<std::mutex> lock(queueMutex);
    
    // Clear the queue
    std::queue<Spike> empty;
    std::swap(spikes, empty);
}

bool SpikeQueue::isEmpty() const
{
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
    return spikes.empty();
}

size_t SpikeQueue::size() const
{
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
    return spikes.size();
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
    }
}

void SpikeQueue::processingLoop()
{
    std::cout << "NSGS: Spike processing thread started" << std::endl;
    
    // Process spikes until terminated
    while (!terminated.load()) {
        Spike spike(-1, -1, 0.0f); // Initialize with default values
        bool hasSpike = tryGetNextSpike(spike);
        
        if (hasSpike) {
            // Process the spike
            if (nodesRef && spike.targetNodeId >= 0 && static_cast<size_t>(spike.targetNodeId) < nodesRef->size()) {
                // Get the target node
                auto& targetNode = (*nodesRef)[spike.targetNodeId];
                
                // Deliver the spike
                targetNode->receiveSpike(spike.weight);
                
                // Increment the processed count
                processedSpikeCount.fetch_add(1);
            }
        } else {
            // No spikes to process, sleep for a short time
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Check thermal adaptation
        checkThermalAdaptation();
    }
    
    std::cout << "NSGS: Spike processing thread terminated" << std::endl;
} 