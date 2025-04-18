/**
 * Data Migration Utility
 * 
 * This script consolidates all dataset files to ensure consistent location
 * and fixes any directory structure issues.
 */

const fs = require('fs');
const path = require('path');

// Base directories
const backendDir = __dirname;
const datasetsDir = path.join(backendDir, 'datasets');

// Make sure datasets directory exists
if (!fs.existsSync(datasetsDir)) {
  fs.mkdirSync(datasetsDir, { recursive: true });
  console.log(`Created datasets directory at ${datasetsDir}`);
}

// List of expected dataset files
const datasetFiles = [
  'imageProcessingData.csv',
  'nsgsAlgorithmData.csv',
  'yolov8AlgorithmData.csv'
];

// Ensure all datasets are in the datasets directory
function migrateDatasets() {
  console.log('Starting dataset migration...');
  
  // Check for CSV files in the backend root directory that should be moved to datasets directory
  datasetFiles.forEach(file => {
    const rootFile = path.join(backendDir, file);
    const datasetFile = path.join(datasetsDir, file);
    
    // Check if file exists in root directory
    if (fs.existsSync(rootFile)) {
      console.log(`Found ${file} in backend root directory`);
      
      // Check if the file already exists in datasets directory
      if (fs.existsSync(datasetFile)) {
        // Compare file sizes to determine which has more data
        const rootStats = fs.statSync(rootFile);
        const datasetStats = fs.statSync(datasetFile);
        
        if (rootStats.size > datasetStats.size) {
          console.log(`Root file ${file} is larger, replacing datasets version`);
          fs.copyFileSync(rootFile, datasetFile);
          fs.unlinkSync(rootFile); // Remove the root file after migration
          console.log(`Removed ${file} from root directory after migration`);
        } else {
          console.log(`Datasets version of ${file} is larger or same size, keeping it`);
          fs.unlinkSync(rootFile); // Remove redundant root file
          console.log(`Removed redundant ${file} from root directory`);
        }
      } else {
        // File doesn't exist in datasets, move it there
        console.log(`Moving ${file} to datasets directory`);
        fs.copyFileSync(rootFile, datasetFile);
        fs.unlinkSync(rootFile); // Remove the root file after migration
        console.log(`Removed ${file} from root directory after migration`);
      }
    }
  });
  
  // Verify all expected dataset files exist in datasets directory
  let allFilesFound = true;
  
  datasetFiles.forEach(file => {
    const filePath = path.join(datasetsDir, file);
    
    if (!fs.existsSync(filePath)) {
      console.log(`Warning: Expected file ${file} not found in datasets directory`);
      allFilesFound = false;
    } else {
      console.log(`Verified: ${file} exists in datasets directory`);
    }
  });
  
  if (allFilesFound) {
    console.log('Migration complete! All expected files found in correct locations');
  } else {
    console.log('Migration complete with warnings. Some expected files not found.');
  }
}

// Run the migration
migrateDatasets(); 