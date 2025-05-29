#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <functional>
#include <memory>
#include <deque>
#include <chrono>
#include <vector>
#include <array>
#include <unordered_map>
#include <condition_variable>

// Forward declaration
class NeuronNode;

// Structure to represent a spike in the queue
struct Spike {
    int sourceNodeId;
    int targetNodeId;
    float weight;
    float priority;   // Higher value = higher priority (edge strength based)
    
    Spike(int source, int target, float w, float p = 0.0f)
        : sourceNodeId(source), targetNodeId(target), weight(w), priority(p) {}
    
    // Default constructor needed for atomic operations
    Spike() : sourceNodeId(-1), targetNodeId(-1), weight(0.0f), priority(0.0f) {}
    
    // Comparison operator for priority queue
    bool operator<(const Spike& other) const {
        return priority < other.priority; // Lower priority value means lower priority
    }
};

// Chase-Lev work-stealing deque implementation
class WorkStealingDeque {
private:
    static const size_t DEQUE_SIZE = 1024; // Must be power of 2
    
    // Circular array for the deque
    std::array<Spike, DEQUE_SIZE> tasks;
    
    // Atomic indices for lock-free operation
    std::atomic<size_t> top;   // Owner pushes/pops from top
    std::atomic<size_t> bottom;// Thieves steal from bottom
    
    // Mask for quick modulo operation (works because DEQUE_SIZE is power of 2)
    static const size_t MASK = DEQUE_SIZE - 1;
    
public:
    WorkStealingDeque() : top(0), bottom(0) {}
    
    // Methods for the owner thread
    void push(const Spike& spike) {
        size_t b = bottom.load(std::memory_order_relaxed);
        tasks[b & MASK] = spike;
        // Memory barrier to ensure task is written before incrementing bottom
        std::atomic_thread_fence(std::memory_order_release);
        bottom.store(b + 1, std::memory_order_relaxed);
    }
    
    bool pop(Spike& result) {
        size_t b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        
        // Memory barrier to ensure bottom is decremented before reading top
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        size_t t = top.load(std::memory_order_relaxed);
        
        if (t <= b) {
            // Deque still has items
            result = tasks[b & MASK];
            
            if (t == b) {
                // Last item contention, use CAS
                if (!top.compare_exchange_strong(t, t + 1, 
                                               std::memory_order_seq_cst,
                                               std::memory_order_relaxed)) {
                    // Failed race with steal operation
                    result = Spike();
                    bottom.store(b + 1, std::memory_order_relaxed);
                    return false;
                }
                bottom.store(b + 1, std::memory_order_relaxed);
            }
            return true;
        } else {
            // Deque is empty
            bottom.store(b + 1, std::memory_order_relaxed);
            return false;
        }
    }
    
    // Method for thief threads
    bool steal(Spike& result) {
        // Memory barrier to ensure correct top/bottom ordering
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        size_t t = top.load(std::memory_order_relaxed);
        
        // Memory barrier before reading bottom
        std::atomic_thread_fence(std::memory_order_acquire);
        
        size_t b = bottom.load(std::memory_order_relaxed);
        
        if (t < b) {
            // Deque has items
            result = tasks[t & MASK];
            
            // Use CAS to ensure only one thief succeeds
            if (!top.compare_exchange_strong(t, t + 1,
                                           std::memory_order_seq_cst,
                                           std::memory_order_relaxed)) {
                // Failed race with another steal or pop
                return false;
            }
            
            return true;
        }
        
        // Deque is empty
        return false;
    }
    
    bool isEmpty() const {
        return bottom.load(std::memory_order_relaxed) <= 
               top.load(std::memory_order_relaxed);
    }
    
    size_t size() const {
        size_t b = bottom.load(std::memory_order_relaxed);
        size_t t = top.load(std::memory_order_relaxed);
        return b > t ? b - t : 0;
    }
};

// Convergence history data structure to track activity over time
struct ConvergenceData {
    size_t queueSize;
    size_t processedSpikes;
    size_t stateChanges;
    std::chrono::steady_clock::time_point timestamp;
    
    ConvergenceData(size_t qs, size_t ps, size_t sc)
        : queueSize(qs), processedSpikes(ps), stateChanges(sc),
          timestamp(std::chrono::steady_clock::now()) {}
};

// Define a hash function for std::pair<int, int>
namespace std {
    template<>
    struct hash<std::pair<int, int>> {
        size_t operator()(const std::pair<int, int>& p) const {
            return hash<int>()(p.first) ^ hash<int>()(p.second);
        }
    };
}

// Thread-safe queue for handling spikes in the NSGS model
class SpikeQueue {
private:
    // Per-thread work-stealing deques
    std::vector<std::unique_ptr<WorkStealingDeque>> threadQueues;
    
    // Global queue for initial distribution - now priority-based
    std::priority_queue<Spike> globalQueue;
    std::mutex globalQueueMutex;
    std::condition_variable globalQueueCondition;
    
    // Atomic counters for queue stats
    std::atomic<size_t> totalQueuedSpikes;
    std::atomic<size_t> processedSpikeCount;
    std::atomic<size_t> queueHighWatermark;
    
    // Control flags
    std::atomic<bool> processing;
    std::atomic<bool> terminated;
    
    // Thread pool for parallel processing
    std::vector<std::thread> workerThreads;
    std::atomic<int> activeWorkers;
    int numThreads;
    
    // Reference to all nodes (for spike propagation)
    std::vector<std::shared_ptr<NeuronNode>>* nodesRef;
    
    // Edge strength maps for priority calculation
    std::unordered_map<int, float> nodeEdgeStrengths;
    std::unordered_map<std::pair<int, int>, float> edgeStrengths;
    std::mutex edgeStrengthMutex;
    
    // Callback for thermal adaptation
    std::function<void(float)> thermalFeedbackCallback;
    
    // Processing statistics
    std::chrono::steady_clock::time_point lastAdaptationTime;
    
    // Convergence tracking
    std::atomic<size_t> stateChangeCount;
    std::deque<ConvergenceData> convergenceHistory;
    std::mutex convergenceHistoryMutex;
    std::chrono::steady_clock::time_point lastActivityTime;
    std::atomic<bool> hasRecentActivity;
    std::atomic<bool> hasConverged;
    
    // Thread synchronization
    std::condition_variable terminationCondition;
    std::mutex terminationMutex;
    
    // Thread ID tracking
    std::atomic<int> nextThreadId;
    thread_local static int threadLocalId;
    
    // Convergence parameters
    const size_t convergenceHistorySize = 10;  // Number of data points to keep
    const float activityThreshold = 0.05f;     // Activity below this % is considered converged
    const int inactivityThresholdMs = 100;     // Consecutive milliseconds without activity for termination
    
    // Worker method for each thread
    void workerThreadFunction(int threadId);
    
    // Process a single spike (used by worker threads)
    void processSingleSpike(const Spike& spike);
    
    // Work stealing functions
    bool getLocalWork(int threadId, Spike& spike);
    bool stealWork(int threadId, Spike& spike);
    bool getGlobalWork(Spike& spike);
    
    // Priority calculation
    float calculateSpikePriority(int sourceNodeId, int targetNodeId, float weight);
    
    // Load balancing
    void balanceLoad();
    
    // Check if any other thread has work (for termination detection)
    bool anyOtherThreadHasWork(int current_thread_id);
    
    // Convergence tracking methods
    void recordConvergenceData();
    bool checkConvergence();
    void updateActivity();
    
    // For robust termination detection
    std::atomic<int> system_idle_confirmations{0};
    std::condition_variable idle_condition_var;
    std::mutex idle_mutex;

public:
    SpikeQueue(int numThreads = 4); // Default to 4 threads
    ~SpikeQueue();
    
    // Queue operations
    void addSpike(const Spike& spike);
    void addSpike(int sourceNodeId, int targetNodeId, float weight);
    
    // Edge strength registration for priority calculation
    void registerNodeEdgeStrength(int nodeId, float edgeStrength);
    void updateNodeEdgeStrength(int nodeId, float edgeStrength);
    
    // Deprecated direct queue access methods - kept for compatibility
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
    size_t getStateChanges() const { return stateChangeCount.load(); }
    bool isConverged() const { return hasConverged.load(); }
    int getActiveWorkers() const { return activeWorkers.load(); }
    int getThreadCount() const { return numThreads; }
    
    // Activity tracking
    void recordStateChange();
    float getConvergenceRate() const;
    std::chrono::milliseconds getTimeSinceLastActivity() const;
    
    // Convergence waiting
    bool waitForConvergence(int maxWaitTimeMs);
    
    // Thermal adaptivity
    void setThermalFeedbackCallback(std::function<void(float)> callback);
    void checkThermalAdaptation();
}; 