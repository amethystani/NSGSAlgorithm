#include "NeuronNode.h"
#include "SpikeQueue.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

NeuronNode::NeuronNode(int id, cv::Point2i position, std::weak_ptr<SpikeQueue> queue)
    : id(id), 
      position(position), 
      statePotential(0.0f),
      threshold(0.5f),
      adaptiveThresholdModifier(1.0f),
      classId(-1),
      confidence(0.0f),
      hasSpike(false),
      refractoryPeriod(0),
      spikeQueue(queue)
{
    // Initialize with empty feature vector
    features.resize(16, 0.0f); // Default size, will be set later
}

void NeuronNode::setFeatures(const std::vector<float>& featureVector)
{
    features = featureVector;
}

void NeuronNode::addConnection(std::weak_ptr<NeuronNode> node, float weight)
{
    connections.push_back(node);
    connectionWeights.push_back(weight);
}

void NeuronNode::resetState()
{
    statePotential.store(0.0f);
    hasSpike = false;
    refractoryPeriod.store(0);
    // Do not reset class ID and confidence as they represent the final state
}

void NeuronNode::incrementPotential(float value)
{
    // Use atomic add operation to safely increment potential
    float current = statePotential.load();
    statePotential.store(current + value);
}

void NeuronNode::decrementPotential(float value)
{
    // Use atomic subtract operation to safely decrement potential
    float current = statePotential.load();
    float newValue = std::max(0.0f, current - value);
    statePotential.store(newValue);
}

void NeuronNode::setThreshold(float value)
{
    threshold = value;
}

void NeuronNode::adaptThreshold(float contextModifier)
{
    adaptiveThresholdModifier = contextModifier;
}

bool NeuronNode::checkAndFire()
{
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
        
        // Only perform consistency check if this node has a class
        if (classId >= 0) {
            // Count active neighbors by class
            std::unordered_map<int, int> neighborClassCounts;
            std::unordered_map<int, float> neighborClassConfidences;
            int activeNeighbors = 0;
            
            for (size_t i = 0; i < connections.size(); i++) {
                if (auto neighbor = connections[i].lock()) {
                    int neighborClass = neighbor->getClassId();
                    if (neighborClass >= 0) {
                        neighborClassCounts[neighborClass]++;
                        neighborClassConfidences[neighborClass] += neighbor->getConfidence();
                        activeNeighbors++;
                    }
                }
            }
            
            // Find dominant class among neighbors
            for (const auto& pair : neighborClassCounts) {
                float avgConfidence = neighborClassConfidences[pair.first] / pair.second;
                if (pair.second > neighborClassCounts[dominantClassId] || 
                    (pair.second == neighborClassCounts[dominantClassId] && 
                     avgConfidence > maxClassConfidence)) {
                    dominantClassId = pair.first;
                    maxClassConfidence = avgConfidence;
                }
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
            // Generate a spike
            hasSpike = true;
            
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
            
            // Propagate spike to neighbors
            propagateToNeighbors();
            
            return true;
        }
    }
    
    return false;
}

void NeuronNode::propagateToNeighbors()
{
    // Get shared_ptr to the queue
    auto queue = spikeQueue.lock();
    if (!queue) return; // Queue not available
    
    // Send spikes to all connected nodes
    for (size_t i = 0; i < connections.size(); i++)
    {
        if (auto neighbor = connections[i].lock())
        {
            // Add spike to the queue (priority calculation happens in SpikeQueue now)
            // The SpikeQueue will automatically prioritize spikes near edges
            queue->addSpike(id, neighbor->getId(), connectionWeights[i]);
        }
    }
}

void NeuronNode::receiveSpike(float weight)
{
    // When receiving a spike, increase potential based on the connection weight
    incrementPotential(weight * 0.5f); // Scale factor to control influence
    
    // NSGS Core: Update class ID based on spikes received
    // If this node has no class yet and the spike is strong enough,
    // check if the source node has a class assigned
    if (classId < 0 && statePotential.load() > threshold * 0.7f) {
        // Try to find the source node in our connections to get its class
        for (size_t i = 0; i < connections.size(); i++) {
            if (auto neighbor = connections[i].lock()) {
                int neighborClassId = neighbor->getClassId();
                if (neighborClassId >= 0) {
                    // Calculate how strongly this node should adopt the neighbor's class
                    // Based on feature similarity and connection strength
                    float similarityScore = 1.0f;
                    const std::vector<float>& myFeatures = getFeatures();
                    const std::vector<float>& neighborFeatures = neighbor->getFeatures();
                    if (!myFeatures.empty() && !neighborFeatures.empty()) {
                        // Simple feature similarity calculation (dot product)
                        float dotProduct = 0.0f;
                        size_t featureSize = std::min(myFeatures.size(), neighborFeatures.size());
                        for (size_t j = 0; j < featureSize; j++) {
                            dotProduct += myFeatures[j] * neighborFeatures[j];
                        }
                        similarityScore = std::max(0.0f, dotProduct / featureSize);
                    }
                    
                    // Class propagation happens when similarity is high and potential is sufficient
                    float propagationStrength = similarityScore * (statePotential.load() / threshold);
                    if (propagationStrength > 0.8f) {
                        classId = neighborClassId;
                        confidence = propagationStrength * neighbor->getConfidence();
                        break;
                    }
                }
            }
        }
    }
    
    // Check if this causes the node to fire
    checkAndFire();
}

cv::Scalar NeuronNode::getColorByState() const
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