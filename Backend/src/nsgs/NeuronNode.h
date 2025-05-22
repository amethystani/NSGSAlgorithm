#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>

// Forward declarations
class SpikeQueue;

class NeuronNode {
private:
    // Node identity and position
    int id;
    cv::Point2i position;
    
    // Node state
    std::atomic<float> statePotential;
    std::atomic<int> refractoryPeriod;
    float threshold;
    float initialThreshold;
    int classId;
    float confidence;
    float adaptiveThresholdModifier;
    float edgeStrength;
    
    // Connectivity
    struct Connection {
        std::shared_ptr<NeuronNode> target;
        float weight;
    };
    std::vector<Connection> connections;
    std::mutex connectionMutex;
    
    // Feature representation
    std::vector<float> features;
    std::mutex nodeMutex;
    
    // Reference to spike queue for event submission
    std::shared_ptr<SpikeQueue> spikeQueue;
    
    // Helper to update effective threshold based on adaptive factors
    void updateEffectiveThreshold();

public:
    // Constructor and destructor
    NeuronNode(int id, cv::Point2i position, std::shared_ptr<SpikeQueue> queue);
    ~NeuronNode() = default;
    
    // Basic getters
    int getId() const { return id; }
    cv::Point2i getPosition() const { return position; }
    float getThreshold() const { return threshold; }
    float getPotential() const { return statePotential.load(); }
    int getClassId() const { return classId; }
    float getConfidence() const { return confidence; }
    const std::vector<float>& getFeatures() const { return features; }
    
    // State management
    void resetState();
    void setThreshold(float value);
    void incrementPotential(float amount);
    void adaptThreshold(float multiplier);
    void setFeatures(const std::vector<float>& featureVector);
    void setClassId(int id) { classId = id; }
    void setConfidence(float value) { confidence = value; }
    void setEdgeStrength(float strength);
    
    // Connectivity
    void addConnection(std::shared_ptr<NeuronNode> target, float weight);
    
    // Spiking behavior
    bool checkAndFire();
    void receiveSpike(float inputStrength, int sourceClassId, float sourceConfidence);
    
    // Visualization helpers
    cv::Scalar getColorForVisualization() const;
}; 