#include "NsgsPredictor.h"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

// Enhanced reconstruction from neural graph using spatial clustering
cv::Mat NsgsPredictor::reconstructFromNeuralGraph()
{
    // Create a blank mask image
    int width = (int)this->inputShapes[0][3];
    int height = (int)this->inputShapes[0][2];
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    
    // Find all nodes with class assignments
    std::unordered_map<int, std::vector<std::shared_ptr<NeuronNode>>> classNodes;
    for (auto &node : graphNodes) {
        int classId = node->getClassId();
        if (classId >= 0) {
            classNodes[classId].push_back(node);
        }
    }
    
    std::cout << "NSGS: Reconstructing segmentation with " << classNodes.size() << " classes" << std::endl;
    
    // For each class, perform spatial clustering of nodes
    for (auto &classPair : classNodes) {
        int classId = classPair.first;
        auto &nodes = classPair.second;
        
        // Skip classes with too few nodes (likely noise)
        if (nodes.size() < 5) {
            std::cout << "NSGS: Skipping class " << classId << " with only " << nodes.size() << " nodes" << std::endl;
            continue;
        }
        
        // Sort nodes by confidence for cluster seeding
        std::sort(nodes.begin(), nodes.end(), [](const std::shared_ptr<NeuronNode> &a, const std::shared_ptr<NeuronNode> &b) {
            return a->getConfidence() > b->getConfidence();
        });
        
        // DBSCAN-like clustering to find coherent regions
        const float distanceThreshold = 24.0f; // Max distance between nodes in same cluster
        const int minClusterSize = 3;         // Minimum nodes for a valid cluster
        
        std::vector<int> nodeClusterIds(nodes.size(), -1);
        int nextClusterId = 0;
        
        for (size_t i = 0; i < nodes.size(); i++) {
            // Skip nodes already assigned to clusters
            if (nodeClusterIds[i] >= 0) continue;
            
            // Start a new cluster from this seed node
            std::queue<size_t> queue;
            queue.push(i);
            nodeClusterIds[i] = nextClusterId;
            
            // BFS expansion to find connected nodes
            while (!queue.empty()) {
                size_t currentIdx = queue.front();
                queue.pop();
                
                cv::Point2i currentPos = nodes[currentIdx]->getPosition();
                
                // Check all other unassigned nodes for potential inclusion
                for (size_t j = 0; j < nodes.size(); j++) {
                    if (nodeClusterIds[j] >= 0) continue; // Skip assigned nodes
                    
                    cv::Point2i otherPos = nodes[j]->getPosition();
                    float distance = cv::norm(currentPos - otherPos);
                    
                    if (distance <= distanceThreshold) {
                        nodeClusterIds[j] = nextClusterId;
                        queue.push(j);
                    }
                }
            }
            
            // Count nodes in this cluster
            int clusterSize = std::count(nodeClusterIds.begin(), nodeClusterIds.end(), nextClusterId);
            
            // Only keep clusters with sufficient size
            if (clusterSize >= minClusterSize) {
                nextClusterId++; // Valid cluster, increment ID for next cluster
            } else {
                // Reset small clusters back to unassigned
                for (size_t j = 0; j < nodeClusterIds.size(); j++) {
                    if (nodeClusterIds[j] == nextClusterId) {
                        nodeClusterIds[j] = -1;
                    }
                }
            }
        }
        
        // For each valid cluster, generate a concave hull and fill it
        for (int clusterId = 0; clusterId < nextClusterId; clusterId++) {
            // Collect points for this cluster
            std::vector<cv::Point> clusterPoints;
            for (size_t i = 0; i < nodes.size(); i++) {
                if (nodeClusterIds[i] == clusterId) {
                    clusterPoints.push_back(nodes[i]->getPosition());
                }
            }
            
            if (clusterPoints.size() < 3) continue; // Need at least 3 points for a polygon
            
            // For very large clusters, skip the concave hull and use a denser fill
            if (clusterPoints.size() > 50) {
                // Draw filled circles for each node with size proportional to local density
                for (const auto& point : clusterPoints) {
                    // Count nearby points to determine local density
                    int nearbyCount = 0;
                    for (const auto& otherPoint : clusterPoints) {
                        if (cv::norm(point - otherPoint) < 16.0f) {
                            nearbyCount++;
                        }
                    }
                    
                    // Radius based on local density
                    int radius = std::min(16, 4 + nearbyCount/2);
                    cv::circle(mask, point, radius, cv::Scalar(classId + 1), -1);
                }
            } else if (clusterPoints.size() >= 3) {
                // For medium clusters, use convex hull as approximation
                std::vector<cv::Point> hull;
                cv::convexHull(clusterPoints, hull);
                
                // Create temporary mask for this cluster
                cv::Mat clusterMask = cv::Mat::zeros(height, width, CV_8UC1);
                std::vector<std::vector<cv::Point>> contours = {hull};
                cv::fillPoly(clusterMask, contours, cv::Scalar(255));
                
                // Copy to output mask with class ID
                mask.setTo(classId + 1, clusterMask);
            }
        }
    }
    
    // Apply morphological operations to clean up the mask
    cv::Mat cleanMask;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, cleanMask, cv::MORPH_CLOSE, element);
    
    return cleanMask;
} 