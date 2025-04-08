#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <functional>
#include <memory>

// Forward declaration
class NeuronNode;

// Structure to represent a spike in the queue
struct Spike {
    int sourceNodeId;
    int targetNodeId;
    float weight;
    
    Spike(int source, int target, float w)
        : sourceNodeId(source), targetNodeId(target), weight(w) {}
};

// Thread-safe queue for handling spikes in the NSGS model
class SpikeQueue {
private:
    std::queue<Spike> spikes;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> processing;
    std::atomic<bool> terminated;
    
    // Processing thread
    std::thread processingThread;
    
    // Reference to all nodes (for spike propagation)
    std::vector<std::shared_ptr<NeuronNode>>* nodesRef;
    
    // Callback for thermal adaptation
    std::function<void(float)> thermalFeedbackCallback;
    
    // Processing statistics
    std::atomic<size_t> processedSpikeCount;
    std::atomic<size_t> queueHighWatermark;
    std::chrono::steady_clock::time_point lastAdaptationTime;
    
    // Worker method to process spikes
    void processingLoop();
    
public:
    SpikeQueue();
    ~SpikeQueue();
    
    // Queue operations
    void addSpike(const Spike& spike);
    void addSpike(int sourceNodeId, int targetNodeId, float weight);
    Spike getNextSpike(); // Blocks until a spike is available
    bool tryGetNextSpike(Spike& spike); // Non-blocking attempt to get a spike
    
    // Queue management
    void startProcessing(std::vector<std::shared_ptr<NeuronNode>>* nodes);
    void stopProcessing();
    void clear();
    
    // Status queries
    bool isProcessing() const { return processing.load(); }
    bool isEmpty() const;
    size_t size() const;
    size_t getProcessedCount() const { return processedSpikeCount.load(); }
    size_t getHighWatermark() const { return queueHighWatermark.load(); }
    
    // Thermal feedback
    void setThermalFeedbackCallback(std::function<void(float)> callback);
    void checkThermalAdaptation();
}; 