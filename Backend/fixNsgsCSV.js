const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

// Path to the NSGS CSV file
const datasetsDir = path.join(__dirname, 'datasets');
const nsgsFilePath = path.join(datasetsDir, 'nsgsAlgorithmData.csv');

// Ensure datasets directory exists
if (!fs.existsSync(datasetsDir)) {
  fs.mkdirSync(datasetsDir, { recursive: true });
  console.log(`Created datasets directory at ${datasetsDir}`);
}

// Define the updated headers with timing columns
const headers = [
  { id: 'timestamp', title: 'Timestamp' },
  { id: 'imageId', title: 'Image ID' },
  { id: 'modelType', title: 'Model Type' },
  { id: 'imageSize', title: 'Image Size (bytes)' },
  { id: 'imageWidth', title: 'Image Width' },
  { id: 'imageHeight', title: 'Image Height' },
  { id: 'totalNodes', title: 'Total Nodes' },
  { id: 'activeNodes', title: 'Active Nodes' },
  { id: 'parallelPathways', title: 'Parallel Pathways' },
  { id: 'executionTime', title: 'Execution Time (ms)' },
  { id: 'standardTime', title: 'Standard Model Time (ms)' },
  { id: 'speedupRatio', title: 'Speedup Ratio' },
  { id: 'memoryUsage', title: 'Memory Usage (MB)' },
  { id: 'cpuUtilization', title: 'CPU Utilization (%)' },
  { id: 'detectedObjects', title: 'Detected Objects' },
  { id: 'parallelizationDepth', title: 'Parallelization Depth' },
  { id: 'graphDensity', title: 'Graph Density' },
  { id: 'averageNodeWeight', title: 'Average Node Weight' },
  { id: 'maxNodeWeight', title: 'Max Node Weight' },
  { id: 'algorithmicEfficiency', title: 'Algorithmic Efficiency' },
  { id: 'threadUtilization', title: 'Thread Utilization (%)' },
  { id: 'neuralSpikes', title: 'Neural Spikes' },
  { id: 'spikeTransmissionRate', title: 'Spike Transmission Rate' },
  { id: 'synchronizationPoints', title: 'Synchronization Points' },
  { id: 'criticalPathLength', title: 'Critical Path Length' },
  { id: 'dependencyResolutions', title: 'Dependency Resolutions' },
  { id: 'concurrentExecutions', title: 'Concurrent Executions' },
  { id: 'resourceUtilizationScore', title: 'Resource Utilization Score' },
  { id: 'loadBalancingEfficiency', title: 'Load Balancing Efficiency (%)' },
  { id: 'neuralGraphComplexity', title: 'Neural Graph Complexity' },
  { id: 'dynamicPathAllocation', title: 'Dynamic Path Allocation' },
  { id: 'bottleneckReduction', title: 'Bottleneck Reduction (%)' },
  { id: 'adaptiveSchedulingEvents', title: 'Adaptive Scheduling Events' },
  { id: 'branchPredictionAccuracy', title: 'Branch Prediction Accuracy (%)' },
  { id: 'memoryAccessPatterns', title: 'Memory Access Patterns' },
  { id: 'inferenceOptimizationLevel', title: 'Inference Optimization Level' },
  { id: 'preprocessTime', title: 'Preprocess Time (ms)' },
  { id: 'inferenceTime', title: 'Inference Time (ms)' },
  { id: 'postprocessTime', title: 'Postprocess Time (ms)' },
  { id: 'ioOverhead', title: 'IO Overhead (ms)' }
];

async function fixNsgsCSV() {
  console.log('Fixing NSGS CSV file...');
  
  // Check if file exists first
  if (fs.existsSync(nsgsFilePath)) {
    // Back up the original file
    const backupPath = `${nsgsFilePath}.backup`;
    fs.copyFileSync(nsgsFilePath, backupPath);
    console.log(`Backed up original CSV to ${backupPath}`);
    
    // Read existing entries
    const existingData = [];
    
    try {
      await new Promise((resolve, reject) => {
        fs.createReadStream(nsgsFilePath)
          .pipe(csv())
          .on('data', (data) => existingData.push(data))
          .on('end', () => {
            console.log(`Read ${existingData.length} entries from existing file`);
            resolve();
          })
          .on('error', (error) => {
            reject(error);
          });
      });
    } catch (error) {
      console.error(`Error reading NSGS CSV: ${error.message}`);
      return;
    }
    
    // Delete the original file
    fs.unlinkSync(nsgsFilePath);
    console.log('Removed original file');
    
    // Create a new file with updated headers
    const csvWriter = createCsvWriter({
      path: nsgsFilePath,
      header: headers
    });
    
    // Prepare updated records with default timing values if missing
    const updatedRecords = existingData.map(record => {
      // Calculate execution time
      const executionTime = parseFloat(record['Execution Time (ms)'] || 0);
      
      // Add timing values if they don't exist
      return {
        ...record,
        // Convert column titles to object property names
        timestamp: record['Timestamp'],
        imageId: record['Image ID'],
        modelType: record['Model Type'],
        imageSize: record['Image Size (bytes)'],
        imageWidth: record['Image Width'],
        imageHeight: record['Image Height'], 
        totalNodes: record['Total Nodes'],
        activeNodes: record['Active Nodes'],
        parallelPathways: record['Parallel Pathways'],
        executionTime: executionTime,
        standardTime: record['Standard Model Time (ms)'],
        speedupRatio: record['Speedup Ratio'],
        memoryUsage: record['Memory Usage (MB)'],
        cpuUtilization: record['CPU Utilization (%)'],
        detectedObjects: record['Detected Objects'],
        parallelizationDepth: record['Parallelization Depth'],
        graphDensity: record['Graph Density'],
        averageNodeWeight: record['Average Node Weight'],
        maxNodeWeight: record['Max Node Weight'],
        algorithmicEfficiency: record['Algorithmic Efficiency'],
        threadUtilization: record['Thread Utilization (%)'],
        neuralSpikes: record['Neural Spikes'],
        spikeTransmissionRate: record['Spike Transmission Rate'],
        synchronizationPoints: record['Synchronization Points'],
        criticalPathLength: record['Critical Path Length'],
        dependencyResolutions: record['Dependency Resolutions'],
        concurrentExecutions: record['Concurrent Executions'],
        resourceUtilizationScore: record['Resource Utilization Score'],
        loadBalancingEfficiency: record['Load Balancing Efficiency (%)'],
        neuralGraphComplexity: record['Neural Graph Complexity'],
        dynamicPathAllocation: record['Dynamic Path Allocation'],
        bottleneckReduction: record['Bottleneck Reduction (%)'],
        adaptiveSchedulingEvents: record['Adaptive Scheduling Events'],
        branchPredictionAccuracy: record['Branch Prediction Accuracy (%)'],
        memoryAccessPatterns: record['Memory Access Patterns'],
        inferenceOptimizationLevel: record['Inference Optimization Level'],
        // Add timing columns with default proportions if missing
        preprocessTime: record['Preprocess Time (ms)'] || (executionTime * 0.15), // 15% of total time
        inferenceTime: record['Inference Time (ms)'] || (executionTime * 0.65),  // 65% of total time
        postprocessTime: record['Postprocess Time (ms)'] || (executionTime * 0.12), // 12% of total time
        ioOverhead: record['IO Overhead (ms)'] || (executionTime * 0.08), // 8% of total time
      };
    });
    
    // Write the updated records
    try {
      await csvWriter.writeRecords(updatedRecords);
      console.log(`Successfully recreated NSGS CSV with ${updatedRecords.length} entries and updated columns`);
    } catch (error) {
      console.error(`Error writing updated NSGS CSV: ${error.message}`);
    }
  } else {
    // If file doesn't exist, just create a new empty one with the correct headers
    console.log('NSGS CSV file does not exist. Creating new file with updated headers.');
    const csvWriter = createCsvWriter({
      path: nsgsFilePath,
      header: headers
    });
    
    await csvWriter.writeRecords([]);
    console.log('Created new empty NSGS CSV file with timing columns');
  }
}

// Run the fix function
fixNsgsCSV()
  .then(() => console.log('NSGS CSV fix completed'))
  .catch(error => console.error(`Error fixing NSGS CSV: ${error.message}`)); 