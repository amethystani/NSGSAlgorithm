const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

class DatasetLogger {
  constructor(logFilePath = null) {
    this.logFilePath = logFilePath || path.join(__dirname, 'imageProcessingData.csv');
    this.headers = [
      { id: 'timestamp', title: 'Timestamp' },
      { id: 'inputFileName', title: 'Input File Name' },
      { id: 'inputFilePath', title: 'Input File Path' },
      { id: 'inputFileSize', title: 'Input File Size (bytes)' },
      { id: 'outputFileName', title: 'Output File Name' },
      { id: 'outputFilePath', title: 'Output File Path' },
      { id: 'outputFileSize', title: 'Output File Size (bytes)' },
      { id: 'modelType', title: 'Model Type' },
      { id: 'nsgsEnabled', title: 'NSGS Enabled' },
      { id: 'processingTimeMs', title: 'Processing Time (ms)' },
      { id: 'detectedObjects', title: 'Detected Objects' },
      { id: 'processedImageUrl', title: 'Processed Image URL' },
      { id: 'metadataPath', title: 'Metadata Path' },
      { id: 'commandExecuted', title: 'Command Executed' }
    ];
    
    this.initializeFile();
  }

  initializeFile() {
    // Check if file exists, if not create it with headers
    if (!fs.existsSync(this.logFilePath)) {
      console.log(`Creating new dataset log file at ${this.logFilePath}`);
      const csvWriter = createCsvWriter({
        path: this.logFilePath,
        header: this.headers
      });
      
      csvWriter.writeRecords([])
        .then(() => console.log('CSV file initialized successfully'))
        .catch(err => console.error(`Error initializing CSV file: ${err.message}`));
    } else {
      console.log(`Dataset log file already exists at ${this.logFilePath}`);
    }
  }

  async logImageProcessing(data) {
    try {
      // Format detected objects as comma-separated list if it's an array
      let detectedObjects = data.detectedObjects;
      if (Array.isArray(detectedObjects)) {
        detectedObjects = detectedObjects.join(', ');
      }

      const record = {
        timestamp: new Date().toISOString(),
        inputFileName: data.inputFileName || '',
        inputFilePath: data.inputFilePath || '',
        inputFileSize: data.inputFileSize || 0,
        outputFileName: data.outputFileName || '',
        outputFilePath: data.outputFilePath || '',
        outputFileSize: data.outputFileSize || 0,
        modelType: data.modelType || '',
        nsgsEnabled: data.nsgsEnabled ? 'true' : 'false',
        processingTimeMs: data.processingTimeMs || 0,
        detectedObjects: detectedObjects || '',
        processedImageUrl: data.processedImageUrl || '',
        metadataPath: data.metadataPath || '',
        commandExecuted: data.commandExecuted || ''
      };

      const csvWriter = createCsvWriter({
        path: this.logFilePath,
        header: this.headers,
        append: true
      });

      await csvWriter.writeRecords([record]);
      console.log(`Successfully logged processing data for ${data.inputFileName}`);
      return true;
    } catch (error) {
      console.error(`Error logging image processing data: ${error.message}`);
      return false;
    }
  }

  async getDataset() {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(this.logFilePath)) {
        return resolve([]);
      }

      const results = [];
      fs.createReadStream(this.logFilePath)
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => {
          resolve(results);
        })
        .on('error', (error) => {
          reject(error);
        });
    });
  }
}

module.exports = new DatasetLogger(); 