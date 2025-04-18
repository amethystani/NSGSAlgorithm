const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

/**
 * YOLOv8 Data Logger
 * Specialized class for logging YOLOv8 standard algorithm performance metrics
 */
class YOLOv8DataLogger {
  constructor() {
    // Store the CSV file in the datasets directory as expected by fixDatabaseSystem.js
    const datasetDir = path.join(__dirname, 'datasets');
    // Ensure the datasets directory exists
    if (!fs.existsSync(datasetDir)) {
      fs.mkdirSync(datasetDir, { recursive: true });
      console.log(`Created datasets directory at ${datasetDir}`);
    }
    this.datasetPath = path.join(datasetDir, 'yolov8AlgorithmData.csv');
    this.initializeDataset();
  }

  /**
   * Initialize the dataset with headers if it doesn't exist
   */
  initializeDataset() {
    if (!fs.existsSync(this.datasetPath)) {
      const csvWriter = createCsvWriter({
        path: this.datasetPath,
        header: [
          { id: 'timestamp', title: 'Timestamp' },
          { id: 'imageId', title: 'Image ID' },
          { id: 'modelType', title: 'Model Type' },
          { id: 'imageSize', title: 'Image Size' },
          { id: 'executionTime', title: 'Execution Time (ms)' },
          { id: 'memoryUsage', title: 'Memory Usage (MB)' },
          { id: 'numObjects', title: 'Detected Objects' },
          { id: 'confidenceAvg', title: 'Average Confidence' },
          { id: 'processingFPS', title: 'Processing FPS' },
          { id: 'numLayers', title: 'Network Layers' },
          { id: 'ioOverhead', title: 'IO Overhead (ms)' },
          { id: 'preprocessTime', title: 'Preprocess Time (ms)' },
          { id: 'inferenceTime', title: 'Inference Time (ms)' },
          { id: 'postprocessTime', title: 'Postprocess Time (ms)' },
        ]
      });

      csvWriter.writeRecords([])
        .then(() => {
          console.log('YOLOv8 dataset initialized successfully');
        })
        .catch(err => {
          console.error('Failed to initialize YOLOv8 dataset:', err);
        });
    }
  }

  /**
   * Estimate YOLOv8 metrics based on image size, processing time, and model type
   * @param {number} imageSize - Size of the image in bytes
   * @param {number} processingTime - Processing time in milliseconds
   * @param {string} modelType - Type of YOLOv8 model (e.g., 'yolov8m', 'yolov8l')
   * @returns {Object} - Estimated metrics
   */
  estimateYolov8Metrics(imageSize, processingTime, modelType = 'yolov8m') {
    // Default values based on model type
    const modelConfigs = {
      'yolov8n': { layers: 168, baseMemory: 3.2, flopsMultiplier: 0.8 },
      'yolov8s': { layers: 225, baseMemory: 11.2, flopsMultiplier: 1.0 },
      'yolov8m': { layers: 295, baseMemory: 25.9, flopsMultiplier: 2.2 },
      'yolov8l': { layers: 365, baseMemory: 43.7, flopsMultiplier: 4.5 },
      'yolov8x': { layers: 476, baseMemory: 68.2, flopsMultiplier: 8.0 },
    };

    // Use default model if provided model not found
    const modelConfig = modelConfigs[modelType] || modelConfigs['yolov8m'];
    
    // Calculate metrics based on image size, processing time, and model type
    const totalNodes = modelConfig.layers;
    const activeNodes = Math.ceil(totalNodes * 0.85); // Estimation: 85% of layers are active during inference
    
    // Memory usage estimate (in MB) - based on model size and image dimensions
    const imageSizeMB = imageSize / (1024 * 1024);
    const memoryUsage = modelConfig.baseMemory + (imageSizeMB * 1.5);
    
    // CPU utilization estimate (percentage)
    const cpuUtilization = Math.min(95, 50 + (processingTime / 50));
    
    // Thread utilization (0-1)
    const threadCount = 4; // Assumption: 4 threads available
    const threadUtilization = Math.min(0.95, 0.4 + (processingTime / (500 * threadCount)));
    
    // FLOPs estimate (Floating Point Operations)
    const baseFlops = 8.0 * (10 ** 9); // 8 GFLOPs base for YOLOv8
    const inferenceFlops = baseFlops * modelConfig.flopsMultiplier;
    
    // Layer timing estimates
    const layerDepth = modelConfig.layers;
    const averageLayerTime = processingTime / layerDepth;
    const maxLayerTime = averageLayerTime * 2.5; // Estimate: max layer takes 2.5x the average
    
    return {
      totalNodes,
      activeNodes,
      memoryUsage,
      cpuUtilization,
      inferenceFlops,
      layerDepth,
      averageLayerTime,
      maxLayerTime,
      threadUtilization
    };
  }

  /**
   * Log YOLOv8 metrics to the dataset
   * @param {Object} metrics - The metrics to log
   * @returns {Promise<boolean>} - Whether the log was successful
   */
  async logYolov8Metrics(metrics) {
    if (!metrics) {
      console.error('Invalid metrics provided for logging');
      return false;
    }

    try {
      const timestamp = new Date().toISOString();
      
      // Calculate processing FPS based on execution time
      const processingFPS = metrics.executionTime > 0 
        ? (1000 / metrics.executionTime).toFixed(2) 
        : 0;
      
      // Calculate confidence average if detectedObjects available
      const confidenceAvg = metrics.confidenceAvg || 
        (metrics.detectedObjects && Array.isArray(metrics.detectedObjects) && metrics.detectedObjects.length > 0
          ? metrics.detectedObjects.reduce((sum, obj) => sum + (obj.confidence || 0), 0) / metrics.detectedObjects.length
          : 0);
      
      // Determine timings
      const inferenceTime = metrics.inferenceTime || (metrics.executionTime * 0.7); // 70% of total time
      const preprocessTime = metrics.preprocessTime || (metrics.executionTime * 0.15); // 15% of total time
      const postprocessTime = metrics.postprocessTime || (metrics.executionTime * 0.1); // 10% of total time
      const ioOverhead = metrics.ioOverhead || (metrics.executionTime * 0.05); // 5% of total time
      
      const entry = {
        timestamp,
        imageId: metrics.imageId || `unknown_${Date.now()}`,
        modelType: metrics.modelType || 'yolov8m',
        imageSize: typeof metrics.imageSize === 'number' ? `${metrics.imageSize} bytes` : metrics.imageSize || '640x640',
        executionTime: metrics.executionTime || 0,
        memoryUsage: metrics.memoryUsage || 0,
        numObjects: Array.isArray(metrics.detectedObjects) ? metrics.detectedObjects.length : (metrics.numObjects || 0),
        confidenceAvg: confidenceAvg,
        processingFPS: processingFPS,
        numLayers: metrics.layerDepth || 0,
        ioOverhead: ioOverhead,
        preprocessTime: preprocessTime,
        inferenceTime: inferenceTime,
        postprocessTime: postprocessTime
      };

      const csvWriter = createCsvWriter({
        path: this.datasetPath,
        append: true,
        header: [
          { id: 'timestamp', title: 'Timestamp' },
          { id: 'imageId', title: 'Image ID' },
          { id: 'modelType', title: 'Model Type' },
          { id: 'imageSize', title: 'Image Size' },
          { id: 'executionTime', title: 'Execution Time (ms)' },
          { id: 'memoryUsage', title: 'Memory Usage (MB)' },
          { id: 'numObjects', title: 'Detected Objects' },
          { id: 'confidenceAvg', title: 'Average Confidence' },
          { id: 'processingFPS', title: 'Processing FPS' },
          { id: 'numLayers', title: 'Network Layers' },
          { id: 'ioOverhead', title: 'IO Overhead (ms)' },
          { id: 'preprocessTime', title: 'Preprocess Time (ms)' },
          { id: 'inferenceTime', title: 'Inference Time (ms)' },
          { id: 'postprocessTime', title: 'Postprocess Time (ms)' },
        ]
      });

      await csvWriter.writeRecords([entry]);
      console.log(`YOLOv8 metrics logged for image ${entry.imageId}`);
      return true;
    } catch (error) {
      console.error(`Error logging YOLOv8 metrics: ${error.message}`);
      return false;
    }
  }

  /**
   * Log a YOLOv8 algorithm data point
   * @param {Object} data - The data point to log
   */
  async logData(data) {
    return this.logYolov8Metrics(data);
  }

  /**
   * Retrieve the entire YOLOv8 dataset
   * @returns {Promise<Array>} The dataset as an array of objects
   */
  async getDataset() {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(this.datasetPath)) {
        this.initializeDataset();
        resolve([]);
        return;
      }

      fs.readFile(this.datasetPath, 'utf8', (err, data) => {
        if (err) {
          console.error(`Error reading YOLOv8 dataset: ${err.message}`);
          reject(err);
          return;
        }

        try {
          const lines = data.trim().split('\n');
          if (lines.length <= 1) {
            resolve([]);
            return;
          }

          const headers = lines[0].split(',');
          const result = [];

          for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue; // Skip empty lines
            
            const values = lines[i].split(',');
            const entry = {};

            for (let j = 0; j < headers.length; j++) {
              entry[headers[j].trim()] = values[j]?.trim() || '';
            }

            result.push(entry);
          }

          resolve(result);
        } catch (error) {
          console.error(`Error parsing YOLOv8 dataset: ${error.message}`);
          reject(error);
        }
      });
    });
  }
}

// Export singleton instance
const yolov8DataLogger = new YOLOv8DataLogger();
module.exports = yolov8DataLogger;

// Create a sample entry to ensure the file is populated properly
if (require.main === module) {
  const sampleData = {
    imageId: 'sample123',
    modelType: 'yolov8m',
    imageSize: '640x640',
    executionTime: 120.5,
    memoryUsage: 850.2,
    numObjects: 5,
    confidenceAvg: 0.86,
    processingFPS: 8.3,
    numLayers: 350,
    ioOverhead: 12.5,
    preprocessTime: 15.3,
    inferenceTime: 85.7,
    postprocessTime: 19.5
  };

  yolov8DataLogger.logData(sampleData)
    .then(() => console.log('Sample YOLOv8 data logged successfully'))
    .catch(err => console.error('Error logging sample data:', err));
} 