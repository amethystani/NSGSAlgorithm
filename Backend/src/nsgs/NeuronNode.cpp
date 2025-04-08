#include "NeuronNode.h"
#include "SpikeQueue.h"
#include <algorithm>
#include <cmath>

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
        // Generate a spike
        hasSpike = true;
        
        // Reset potential
        statePotential.store(0.0f);
        
        // Enter refractory period (5 cycles as an example)
        refractoryPeriod.store(5);
        
        // Propagate spike to neighbors
        propagateToNeighbors();
        
        return true;
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
            // Add spike to the queue
            queue->addSpike(id, neighbor->getId(), connectionWeights[i]);
        }
    }
}

void NeuronNode::receiveSpike(float weight)
{
    // When receiving a spike, increase potential based on the connection weight
    incrementPotential(weight * 0.5f); // Scale factor to control influence
    
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