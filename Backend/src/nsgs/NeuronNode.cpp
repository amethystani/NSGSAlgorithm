#include "NeuronNode.h"
#include "SpikeQueue.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <vector>

NeuronNode::NeuronNode(int id, cv::Point2i position, std::shared_ptr<SpikeQueue> queue)
    : id(id), 
      position(position), 
      statePotential(0.0f),
      refractoryPeriod(0),
      threshold(0.5f),
      initialThreshold(0.5f),
      classId(-1),
      confidence(0.0f),
      adaptiveThresholdModifier(1.0f),
      edgeStrength(0.0f),
      spikeQueue(queue)
{
    // Initialize with empty feature vector
    features.resize(16, 0.0f); // Default size, will be set later
}

void NeuronNode::setFeatures(const std::vector<float>& featureVector)
{
    // Safety check for very large feature vectors
    if (featureVector.size() > 10000) { // Arbitrary reasonable limit to prevent memory issues
        std::cerr << "NeuronNode: Feature vector too large: " << featureVector.size() << " elements" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(nodeMutex);
    features = featureVector;
}

void NeuronNode::addConnection(std::shared_ptr<NeuronNode> target, float weight)
{
    // Safety check for target node
    if (!target) {
        std::cerr << "NeuronNode: Attempt to add null connection target" << std::endl;
        return;
    }
    
    // Safety check for valid weight values
    if (weight <= 0.0f || std::isnan(weight) || std::isinf(weight)) {
        std::cerr << "NeuronNode: Invalid connection weight: " << weight << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(connectionMutex);
    
    // Add connection to the target node with specified weight
    connections.push_back({target, weight});
}

void NeuronNode::resetState()
{
    statePotential.store(0.0f);
    refractoryPeriod.store(0);
    // Do not reset class ID and confidence as they represent the final state
}

void NeuronNode::incrementPotential(float amount)
{
    // Safety check for valid increment
    if (amount <= 0.0f || std::isnan(amount) || std::isinf(amount)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(nodeMutex);
    float current = statePotential.load();
    float newValue = current + amount;
    
    // Cap potential at a maximum value to prevent numerical issues
    const float maxPotential = 10.0f;
    newValue = std::min(newValue, maxPotential);
    
    statePotential.store(newValue);
}

void NeuronNode::setThreshold(float value)
{
    threshold = value;
    initialThreshold = value;
}

void NeuronNode::adaptThreshold(float multiplier)
{
    // Safety check for valid multiplier
    if (multiplier <= 0.0f || std::isnan(multiplier) || std::isinf(multiplier)) {
        return;
    }
    
    // Clamp multiplier to reasonable range
    float safeMultiplier = std::min(5.0f, std::max(0.2f, multiplier));
    
    std::lock_guard<std::mutex> lock(nodeMutex);
    adaptiveThresholdModifier = safeMultiplier;
    
    // Recalculate effective threshold with adaptation factors
    updateEffectiveThreshold();
}

bool NeuronNode::checkAndFire()
{
    // Temporary store for spikes to be added outside the lock
    std::vector<Spike> spikes_to_add; // Spike struct from SpikeQueue.h

    // Scope for nodeMutex lock
    {
        std::lock_guard<std::mutex> lock(nodeMutex);
        
        // If in refractory period, cannot fire
        if (refractoryPeriod.load() > 0)
        {
            refractoryPeriod.fetch_sub(1);
            return false;
        }
        
        // Check if potential exceeds threshold
        float currentPotential = statePotential.load();
        float effectiveThreshold = threshold * adaptiveThresholdModifier;

        
        if (currentPotential > effectiveThreshold)
        {
            // NSGS Enhancement: Check neighborhood class consistency before firing
            // This helps form clearer boundaries between segments
            bool shouldFire = true;
            int dominantClassId = -1;
            float maxClassConfidence = 0.0f;
            std::unordered_map<int, int> neighborClassCounts;
            
            // Only perform consistency check if this node has a class
            if (classId >= 0) {
                // Count active neighbors by class
                std::unordered_map<int, float> neighborClassConfidences;
                int activeNeighbors = 0;
                
                for (size_t i = 0; i < connections.size(); i++) {
                    if (connections[i].target) {
                        int neighborClass = connections[i].target->getClassId();
                        if (neighborClass >= 0) {
                            neighborClassCounts[neighborClass]++;
                            neighborClassConfidences[neighborClass] += connections[i].target->getConfidence();
                            activeNeighbors++;
                        }
                    }
                }
                
                // Find dominant class among neighbors
                if (!neighborClassCounts.empty()) {
                    dominantClassId = neighborClassCounts.begin()->first;
                    maxClassConfidence = 0.0f;
                    for (const auto& pair : neighborClassCounts) {
                        float avgConfidence = (pair.second > 0) ? neighborClassConfidences[pair.first] / pair.second : 0.0f;
                        auto dominantEntryIt = neighborClassCounts.find(dominantClassId);
                        int dominantClassCount = (dominantEntryIt != neighborClassCounts.end()) ? dominantEntryIt->second : 0;

                        if (pair.second > dominantClassCount || 
                            (pair.second == dominantClassCount && avgConfidence > maxClassConfidence)) {
                            dominantClassId = pair.first;
                            maxClassConfidence = avgConfidence;
                        }
                    }
                } else {
                    dominantClassId = -1;
                }

                
                // If this node's class differs from dominant neighbor class and 
                // the dominant class has strong representation, inhibit firing
                // This prevents crossing established segment boundaries
                if (activeNeighbors > 3 && // At least 3 classified neighbors
                    dominantClassId >= 0 && dominantClassId != classId && 
                    neighborClassCounts[dominantClassId] > activeNeighbors/2) {
                    
                    // Instead of firing, adapt class to match neighborhood if confidence is low
                    // This helps clean up noisy classifications
                    if (confidence < 0.6f && maxClassConfidence > 0.7f) {
                        classId = dominantClassId;
                        confidence = 0.8f * maxClassConfidence; // Slightly reduce confidence for propagated classes
                        
                        // Reduce potential to prevent immediate firing with new class
                        statePotential.store(0.5f * effectiveThreshold);
                        return false;
                    }
                    
                    // High confidence nodes resist class changes and don't fire across boundaries
                    if (confidence > 0.7f) {
                        shouldFire = false;
                        // Strong class identity enhances refractory period
                        refractoryPeriod.store(8);
                        // And reduces potential
                        statePotential.store(0.25f * effectiveThreshold);
                        return false;
                    }
                }
            }
            
            if (shouldFire) {
                // Reset potential
                statePotential.store(0.0f);
                
                // Enter refractory period - higher for nodes with assigned classes
                int baseRefractoryPeriod = 5;
                if (classId >= 0) {
                    // Classified nodes have longer refractory periods proportional to confidence
                    refractoryPeriod.store(baseRefractoryPeriod + static_cast<int>(5.0f * confidence));
                } else {
                    refractoryPeriod.store(baseRefractoryPeriod);
                }
                
                // Collect spikes to be added
                if (spikeQueue) {
                    for (size_t i = 0; i < connections.size(); i++) {
                        if (connections[i].target) {
                            // Add spike details for later processing
                            spikes_to_add.emplace_back(id, connections[i].target->getId(), connections[i].weight, 0.0f);
                        }
                    }
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    } // nodeMutex is released here

    // Add collected spikes to the queue outside the lock
    if (!spikes_to_add.empty() && spikeQueue) {
        for (const auto& spike_data : spikes_to_add) {
            spikeQueue->addSpike(spike_data.sourceNodeId, spike_data.targetNodeId, spike_data.weight);
        }
        return true;
    }
    
    return false;
}

void NeuronNode::receiveSpike(float inputStrength, int sourceClassId, float sourceConfidence)
{
    // Safety check for invalid inputs
    if (inputStrength <= 0.0f || sourceClassId < 0 || sourceConfidence <= 0.0f) {
        return;
    }
    
    float adaptedStrength = inputStrength;
    
    std::lock_guard<std::mutex> lock(nodeMutex);
    
    // Increment membrane potential
    statePotential.store(statePotential.load() + adaptedStrength);
    
    // Class competition (only adopt source class if it has higher confidence than current)
    if (sourceClassId >= 0 && (classId < 0 || sourceConfidence > confidence)) {
        classId = sourceClassId;
        
        // Slightly reduce confidence with each hop for more conservative propagation
        confidence = sourceConfidence * 0.95f;
    }
    
    // Check if this causes the neuron to fire
    // Note: not calling checkAndFire() here to avoid deadlock (we'd need to release nodeMutex)
    // The node will be checked in the next processing cycle
}

cv::Scalar NeuronNode::getColorForVisualization() const
{
    if (classId >= 0)
    {
        // If assigned to a class, use a color based on the class ID
        // Simple HSV to RGB conversion for distinct colors
        float hue = (classId * 30) % 180;
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Vec3b color = rgb.at<cv::Vec3b>(0, 0);
        return cv::Scalar(color[0], color[1], color[2]);
    }
    else
    {
        // If not assigned, color based on potential
        float potentialRatio = statePotential.load() / (threshold * adaptiveThresholdModifier);
        potentialRatio = std::min(1.0f, potentialRatio);
        
        // Blue to red gradient based on potential
        int red = static_cast<int>(255 * potentialRatio);
        int blue = static_cast<int>(255 * (1.0f - potentialRatio));
        return cv::Scalar(blue, 0, red);
    }
}

void NeuronNode::updateEffectiveThreshold()
{
    // Recalculate effective threshold with adaptation factors
    float newThreshold = threshold * adaptiveThresholdModifier;
    statePotential.store(std::min(statePotential.load(), newThreshold));
}

void NeuronNode::setEdgeStrength(float strength)
{
    // Safety check for valid strength value
    if (strength < 0.0f || std::isnan(strength) || std::isinf(strength)) {
        return;
    }
    
    // Clamp to valid range
    float safeStrength = std::min(1.0f, std::max(0.0f, strength));
    
    std::lock_guard<std::mutex> lock(nodeMutex);
    edgeStrength = safeStrength;
    
    // Edge neurons (high edge strength) should have higher thresholds
    // to prevent unwanted propagation across object boundaries
    updateEffectiveThreshold();
} 