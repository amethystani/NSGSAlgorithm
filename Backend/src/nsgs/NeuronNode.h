#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>

// Forward declaration
class SpikeQueue;

class NeuronNode {
private:
    int id;                                 // Unique identifier for this node
    cv::Point2i position;                   // Position in image coordinates
    std::atomic<float> statePotential;      // Current activation potential (atomic for thread safety)
    float threshold;                         // Firing threshold
    float adaptiveThresholdModifier;        // Adaptive threshold based on local context
    
    // Feature vector for this node (extracted from image or model embeddings)
    std::vector<float> features;
    
    // Segmentation related properties
    int classId;                            // Assigned class ID (-1 if not assigned)
    float confidence;                       // Confidence in the classification
    
    // Spiking state
    bool hasSpike;                          // Whether this node has a spike ready to propagate
    std::atomic<int> refractoryPeriod;      // Refractory period counter (atomic for thread safety)
    
    // Connected nodes (spatial and feature-based connections)
    std::vector<std::weak_ptr<NeuronNode>> connections;
    std::vector<float> connectionWeights;    // Weights for each connection
    
    // Reference to global spike queue (for propagating signals)
    std::weak_ptr<SpikeQueue> spikeQueue;
    
public:
    NeuronNode(int id, cv::Point2i position, std::weak_ptr<SpikeQueue> queue);
    
    // Node initialization
    void setFeatures(const std::vector<float>& featureVector);
    void addConnection(std::weak_ptr<NeuronNode> node, float weight);
    
    // State management
    void resetState();
    void incrementPotential(float value);
    void decrementPotential(float value);
    
    // Threshold management
    void setThreshold(float value);
    void adaptThreshold(float contextModifier);
    
    // Spike processing
    bool checkAndFire();                     // Check if threshold is exceeded and fire if needed
    void propagateToNeighbors();             // Propagate spike to connected nodes
    void receiveSpike(float weight);         // Receive a spike from another node
    
    // Getters and setters
    int getId() const { return id; }
    cv::Point2i getPosition() const { return position; }
    float getPotential() const { return statePotential.load(); }
    float getThreshold() const { return threshold * adaptiveThresholdModifier; }
    int getClassId() const { return classId; }
    float getConfidence() const { return confidence; }
    const std::vector<float>& getFeatures() const { return features; }
    
    void setClassId(int id) { classId = id; }
    void setConfidence(float conf) { confidence = conf; }
    
    // Used for visualization and debugging
    cv::Scalar getColorByState() const;
}; 