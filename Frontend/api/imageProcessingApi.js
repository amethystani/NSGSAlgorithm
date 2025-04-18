import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import { Platform } from 'react-native';

// Determine the correct API URL based on platform
// Use the specific IP address for all devices when testing with Expo Go
export const getApiUrl = () => {
  // Use localhost for web and for local testing
  if (Platform.OS === 'web') {
    console.log('Using web platform, API URL will be localhost');
    return 'http://localhost:3000';
  }
  
  // For devices (iOS/Android), use the network IP
  // This is the actual IP address of your machine on the network
  const DEVICE_IP = '10.79.107.65'; // Verified this IP matches your current network configuration
  
  console.log(`Using mobile platform (${Platform.OS}), API URL will be http://${DEVICE_IP}:3000`);
  return `http://${DEVICE_IP}:3000`;
};

const API_URL = getApiUrl();
console.log(`Using API URL: ${API_URL}`);
console.log(`Platform: ${Platform.OS}, API URL: ${API_URL}`);

// Helper function to create a fetch request with timeout
const fetchWithTimeout = async (url, options = {}, timeout = 120000, retries = 3) => {
  const controller = new AbortController();
  const signal = controller.signal;
  
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  let lastError;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        signal
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      lastError = error;
      console.log(`Fetch attempt ${attempt + 1}/${retries} failed:`, error.message);
      
      // Don't retry if we aborted deliberately
      if (error.name === 'AbortError') {
        clearTimeout(timeoutId);
        throw new Error(`Request timed out after ${timeout}ms`);
      }
      
      // Wait a bit before retrying
      if (attempt < retries - 1) {
        const delay = 1000 * Math.pow(2, attempt); // Exponential backoff
        console.log(`Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  clearTimeout(timeoutId);
  throw lastError || new Error('Failed to fetch after multiple attempts');
};

/**
 * Test the API connection
 * @returns {Promise<boolean>} - True if the API is accessible
 */
export const testApiConnection = async () => {
  try {
    console.log(`Testing connection to API at: ${API_URL}`);
    
    const response = await fetchWithTimeout(`${API_URL}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    }, 10000); // 10 second timeout for API test
    
    if (!response.ok) {
      console.error(`API test failed with status: ${response.status}`);
      return false;
    }
    
    const data = await response.json();
    console.log('API connection successful:', data);
    return true;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
  }
};

/**
 * Debug upload an image to test connection without processing
 * @param {string} imageUri - Local URI of the image to upload
 * @returns {Promise<Object>} - Result with uploaded file information
 */
export const debugUploadImage = async (imageUri) => {
  try {
    console.log(`Debug uploading image: ${imageUri}`);
    
    // Get the filename from the URI
    const filename = imageUri.split('/').pop() || 'image.jpg';
    console.log(`Using filename: ${filename}`);
    
    // Create FormData
    const formData = new FormData();
    
    // Check if we're running on web
    if (Platform.OS === 'web') {
      console.log('Web platform detected, using direct fetch with blob');
      
      try {
        // For web, we need to fetch the image as a blob
        const response = await fetch(imageUri);
        const blob = await response.blob();
        
        formData.append('image', blob, filename);
        
        console.log(`Sending web fetch request to: ${API_URL}/debug-upload`);
        
        const result = await fetchWithTimeout(`${API_URL}/debug-upload`, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          }
        }, 30000).then(res => {
          if (!res.ok) {
            return res.text().then(text => {
              throw new Error(text || `Server responded with status ${res.status}`);
            });
          }
          return res.json();
        });
        
        console.log('Debug upload result:', result);
        return result;
      } catch (error) {
        console.error('Error in web debug upload:', error);
        throw error;
      }
    } else {
      // Native platforms (iOS/Android)
      console.log('Native platform detected, using ImageManipulator');
      
      // For React Native, we need to get the actual image data as base64
      let imageBase64;
      try {
        // First, compress the image more aggressively for large images
        const manipulateResult = await ImageManipulator.manipulateAsync(
          imageUri,
          [{ resize: { width: 800 } }],
          { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG, base64: true }
        );
        
        imageBase64 = manipulateResult.base64;
        console.log(`Read base64 data with length: ${imageBase64.length}`);
      } catch (readError) {
        console.error('Error reading image file:', readError);
        throw new Error(`Could not read image file: ${readError.message}`);
      }
      
      if (!imageBase64 || imageBase64.length === 0) {
        throw new Error('Failed to read image data as base64');
      }
      
      // Get file extension for mime type
      const ext = filename.split('.').pop()?.toLowerCase() || 'jpg';
      const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
      
      // Add the image as base64 string
      formData.append('imageBase64', imageBase64);
      formData.append('filename', filename);
      formData.append('mimeType', mimeType);
      
      console.log(`Sending native request to: ${API_URL}/debug-upload with base64 data`);
      
      // Use fetch with the base64 data
      const response = await fetchWithTimeout(`${API_URL}/debug-upload`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        }
      }, 30000);
      
      if (!response.ok) {
        const text = await response.text();
        console.error(`Server responded with ${response.status}: ${text}`);
        throw new Error(text || `Server responded with status ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Debug upload result:', result);
      return result;
    }
  } catch (error) {
    console.error('Error in debug upload:', error);
    throw error;
  }
};

// Define the ProcessedImage type
/**
 * @typedef {Object} ProcessedImage
 * @property {string} name - The name of the image
 * @property {string} url - The URL of the processed image
 * @property {number} [processingTime] - The time taken to process the image in seconds
 */

// Define the ProcessedImageResult type
/**
 * @typedef {Object} ProcessedImageResult
 * @property {string} [fullUrl] - The full URL of the processed image
 * @property {string} [processedImageUrl] - The relative URL of the processed image
 * @property {boolean} [isFallback] - Whether the result is a fallback
 * @property {string} [warning] - A warning message if there was an issue
 * @property {string} [message] - A message about the processing
 * @property {number} [processingTime] - The time taken to process the image in seconds
 */

/**
 * @typedef {Object} DebugUploadResult
 * @property {Object} [file] - Information about the uploaded file
 * @property {string} [file.originalname] - The original name of the file
 * @property {number} [file.size] - The size of the file in bytes
 */

/**
 * Process an image with YOLOv8
 * @param {string} imageUri - Local URI of the image to process
 * @param {string} modelType - The model to use (yolov8m or yolov8m-seg)
 * @param {boolean} useNSGS - Whether to use the NSGS approach
 * @returns {Promise<ProcessedImageResult>} - Result with processed image URL
 */
export const processImage = async (imageUri, modelType = 'yolov8m-seg', useNSGS = false) => {
  try {
    // Ensure useNSGS is a boolean
    const nsgsEnabled = useNSGS === true;
    
    console.log(`Processing image with model: ${modelType}, NSGS: ${nsgsEnabled ? 'Enabled ✓' : 'Disabled ✗'}`);
    console.log(`NSGS flag: ${nsgsEnabled} (${typeof nsgsEnabled})`);
    
    if (nsgsEnabled) {
      console.log('Using NSGS (Neuro-Scheduling for Graph Segmentation) for optimized parallel processing');
    } else {
      console.log('Using STANDARD processing without NSGS optimization');
    }
    
    // Get the filename from the URI
    const filename = imageUri.split('/').pop() || 'image.jpg';
    console.log(`Using filename: ${filename}`);
    
    // Create FormData
    const formData = new FormData();
    
    // Check if we're running on web
    if (Platform.OS === 'web') {
      console.log('Web platform detected, using direct fetch with blob');
      
      try {
        // For web, we need to fetch the image as a blob
        const response = await fetch(imageUri);
        const blob = await response.blob();
        
        formData.append('image', blob, filename);
        formData.append('modelType', modelType);
        // Very explicitly set the useNSGS value with extra logging
        console.log(`About to append useNSGS (web): ${nsgsEnabled} (${typeof nsgsEnabled})`);
        const nsgsStringValue = nsgsEnabled === true ? 'true' : 'false';
        console.log(`String value to send (web): "${nsgsStringValue}"`);
        formData.append('useNSGS', nsgsStringValue);
        
        console.log(`Sending web fetch request to: ${API_URL}/process`);
        console.log(`Using NSGS optimization: ${nsgsEnabled ? 'YES' : 'NO'}`);
        
        const result = await fetchWithTimeout(`${API_URL}/process`, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          }
        }, 180000).then(res => {
          if (!res.ok) {
            return res.text().then(text => {
              throw new Error(text || `Server responded with status ${res.status}`);
            });
          }
          return res.json();
        });
        
        console.log('Process result:', result);
        return result;
      } catch (error) {
        console.error('Error in web processing:', error);
        throw error;
      }
    } else {
      // Native platforms (iOS/Android)
      console.log('Native platform detected');
      
      // Possible scenarios:
      // 1. Image is a local file (file://)
      // 2. Image is a base64 string
      // 3. Image is a remote URL
      
      if (imageUri.startsWith('data:')) {
        // This is already a base64 string
        console.log('Image is a base64 string');
        const base64Data = imageUri.split(',')[1];
        
        formData.append('imageBase64', base64Data);
        formData.append('filename', filename);
        formData.append('modelType', modelType);
        formData.append('useNSGS', nsgsEnabled ? 'true' : 'false');
        
        // Get file extension from mime type in data URI
        const mimeType = imageUri.split(',')[0].split(':')[1].split(';')[0];
        formData.append('mimeType', mimeType);
      }
      else {
        try {
          // Try to read the file as base64
          let imageBase64;
          
          // First check if we need to resize for performance
          try {
            // First, compress the image more aggressively for large images
            const manipulateResult = await ImageManipulator.manipulateAsync(
              imageUri,
              [{ resize: { width: 800 } }],
              { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG, base64: true }
            );
            
            imageBase64 = manipulateResult.base64;
            console.log(`Read base64 data with length: ${imageBase64.length}`);
          } catch (readError) {
            console.error('Error processing image file:', readError);
            throw new Error(`Could not process image file: ${readError.message}`);
          }
          
          if (!imageBase64 || imageBase64.length === 0) {
            throw new Error('Failed to read image data as base64');
          }
          
          // Get file extension for mime type
          const ext = filename.split('.').pop()?.toLowerCase() || 'jpg';
          const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
          
          // Add the image as base64 string
          formData.append('imageBase64', imageBase64);
          formData.append('filename', filename);
          formData.append('mimeType', mimeType);
          formData.append('modelType', modelType);
          // Very explicitly set the useNSGS value with extra logging
          console.log(`About to append useNSGS: ${nsgsEnabled} (${typeof nsgsEnabled})`);
          const nsgsStringValue = nsgsEnabled === true ? 'true' : 'false';
          console.log(`String value to send: "${nsgsStringValue}"`);
          formData.append('useNSGS', nsgsStringValue);
          
          // Send the request to the server
          console.log(`Sending request to: ${API_URL}/process with base64 data`);
          console.log(`Using NSGS optimization: ${nsgsEnabled ? 'YES' : 'NO'}`);
          
          const response = await fetchWithTimeout(`${API_URL}/process`, {
            method: 'POST',
            body: formData,
            headers: {
              'Accept': 'application/json',
            }
          }, 180000);
          
          if (!response.ok) {
            const text = await response.text();
            console.error(`Server responded with ${response.status}: ${text}`);
            throw new Error(text || `Server responded with status ${response.status}`);
          }
          
          const result = await response.json();
          console.log('Processing result:', result);
          
          // For mobile platforms, download the image to local cache storage
          if (Platform.OS !== 'web' && (result.processedImageUrl || result.fullUrl)) {
            try {
              // Get the full URL with the API base URL
              const imgUrl = result.fullUrl || `${API_URL}${result.processedImageUrl}`;
              
              // Ensure the URL has correct protocol and host
              const fullUrl = imgUrl.startsWith('http') 
                ? imgUrl
                : (imgUrl.startsWith('/') ? `${API_URL}${imgUrl}` : `${API_URL}/${imgUrl}`);
              
              console.log('Downloading processed image to local cache:', fullUrl);
              
              // Create cache directory if it doesn't exist
              const cacheDir = `${FileSystem.cacheDirectory}processed-images/`;
              const cacheDirInfo = await FileSystem.getInfoAsync(cacheDir);
              if (!cacheDirInfo.exists) {
                await FileSystem.makeDirectoryAsync(cacheDir, { intermediates: true });
                console.log('Created cache directory:', cacheDir);
              }
              
              // Generate local filename with timestamp to avoid cache issues
              const timestamp = Date.now();
              const localFilename = `${timestamp}_${result.processedImageName || 'processed.jpg'}`;
              const localUri = `${cacheDir}${localFilename}`;
              
              // Download the file with retry logic
              console.log(`Downloading image from ${fullUrl} to ${localUri}`);
              
              // Try up to 3 times with increasing delays
              let downloadSuccess = false;
              let downloadError = null;
              
              for (let attempt = 0; attempt < 3 && !downloadSuccess; attempt++) {
                try {
                  if (attempt > 0) {
                    console.log(`Retry attempt ${attempt + 1}/3...`);
                    // Wait before retrying (exponential backoff)
                    await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
                  }
                  
                  // Add cache-busting parameter to URL to avoid cached responses
                  const cacheBustUrl = fullUrl.includes('?') 
                    ? `${fullUrl}&t=${timestamp}` 
                    : `${fullUrl}?t=${timestamp}`;
                  
                  console.log(`Using URL with cache-busting: ${cacheBustUrl}`);
                  
                  // Use a longer timeout for downloads
                  const downloadOptions = {
                    md5: false,
                    headers: {
                      'Cache-Control': 'no-cache',
                      'Pragma': 'no-cache'
                    }
                  };
                  
                  const downloadResult = await FileSystem.downloadAsync(cacheBustUrl, localUri, downloadOptions);
                  
                  // Verify the download was successful
                  if (downloadResult.status === 200) {
                    console.log('Download reported success, verifying file...');
                    
                    // Verify the file exists and has content
                    const fileInfo = await FileSystem.getInfoAsync(localUri);
                    if (fileInfo.exists && fileInfo.size > 0) {
                      console.log(`Downloaded file size: ${fileInfo.size} bytes`);
                      downloadSuccess = true;
                      
                      // Add the local URI to the result
                      result.localUri = localUri;
                      result.cachedImageUrl = `file://${localUri}`;
                      
                      console.log('Added local URI to result:', result.cachedImageUrl);
                      
                      // Try to read a few bytes to ensure the file is valid
                      try {
                        const head = await FileSystem.readAsStringAsync(localUri, { 
                          encoding: FileSystem.EncodingType.Base64,
                          position: 0,
                          length: 100
                        });
                        if (head && head.length > 0) {
                          console.log('File content verification successful');
                        } else {
                          console.warn('File exists but might be empty or corrupted');
                          // Continue with the cached file anyway
                        }
                      } catch (readError) {
                        console.warn('Could not verify file content, might be corrupted:', readError);
                        // Continue with the cached file anyway
                      }
                    } else {
                      console.warn(`Downloaded file issue: exists=${fileInfo.exists}, size=${fileInfo.size || 'unknown'}`);
                      downloadError = new Error('Downloaded file is empty or does not exist');
                    }
                  } else {
                    console.warn(`Download returned status: ${downloadResult.status}`);
                    downloadError = new Error(`Download failed with status: ${downloadResult.status}`);
                  }
                } catch (attemptError) {
                  console.error(`Download attempt ${attempt + 1} failed:`, attemptError);
                  downloadError = attemptError;
                }
              }
              
              if (!downloadSuccess) {
                // If all download attempts failed, try to use base64 encoding directly
                console.log('All download attempts failed. Trying to get image directly as base64...');
                
                try {
                  // Try to fetch the image directly and convert to base64 rather than saving to file
                  const response = await fetch(fullUrl, {
                    headers: {
                      'Cache-Control': 'no-cache',
                      'Pragma': 'no-cache'
                    }
                  });
                  
                  if (response.ok) {
                    const blob = await response.blob();
                    const reader = new FileReader();
                    
                    // Convert blob to base64
                    const base64 = await new Promise((resolve, reject) => {
                      reader.onload = () => resolve(reader.result);
                      reader.onerror = reject;
                      reader.readAsDataURL(blob);
                    });
                    
                    if (base64) {
                      console.log('Successfully fetched image as base64');
                      result.imageBase64 = base64;
                      downloadSuccess = true;
                    }
                  }
                } catch (base64Error) {
                  console.error('Failed to fetch image as base64:', base64Error);
                }
                
                if (!downloadSuccess) {
                  console.error('imageprocessingapi.js: all download attempts failed', downloadError);
                  // Continue with the remote URL - we'll fall back to it
                }
              }
            } catch (downloadError) {
              console.error('Error downloading image to local cache imageprocessingapi.js:', downloadError);
              // Continue even if download fails - we'll try to use the remote URL
            }
          }
          
          return result;
        } catch (error) {
          console.error('Error processing image:', error);
          throw error;
        }
      }
    }
  } catch (error) {
    console.error('Error processing image:', error);
    throw error;
  }
};

/**
 * Get the list of processed images from the backend
 * @returns {Promise<Array>} - Array of processed image objects
 */
export const getProcessedImages = async () => {
  try {
    console.log(`Fetching processed images from: ${API_URL}/history`);
    
    const response = await fetchWithTimeout(`${API_URL}/history`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      }
    }, 15000);
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error(`Failed to fetch processed images: ${JSON.stringify(errorData)}`);
      throw new Error(errorData.error || 'Failed to fetch processed images');
    }
    
    const data = await response.json();
    console.log(`Successfully fetched ${data.images.length} processed images`);
    
    // Transform URLs to include the full API URL
    return data.images.map(image => ({
      ...image,
      url: `${API_URL}${image.url}`
    }));
  } catch (error) {
    console.error('Error fetching processed images:', error);
    throw error;
  }
};

/**
 * Download a processed image to the device
 * @param {string} imageUrl - URL of the image to download
 * @param {string} filename - Name to save the file as
 * @returns {Promise<string>} - Local URI of the downloaded file
 */
export const downloadProcessedImage = async (imageUrl, filename) => {
  try {
    console.log(`Downloading image from: ${imageUrl}`);
    
    const downloadDirectory = FileSystem.documentDirectory + 'downloads/';
    
    // Ensure the download directory exists
    const dirInfo = await FileSystem.getInfoAsync(downloadDirectory);
    if (!dirInfo.exists) {
      await FileSystem.makeDirectoryAsync(downloadDirectory, { intermediates: true });
    }
    
    const fileUri = downloadDirectory + filename;
    
    // If the URL doesn't start with http, add the API_URL
    const fullImageUrl = imageUrl.startsWith('http') 
      ? imageUrl 
      : `${API_URL}${imageUrl}`;
    
    console.log(`Full download URL: ${fullImageUrl}`);
    console.log(`Will save to: ${fileUri}`);
    
    // Download the file
    const downloadResult = await FileSystem.downloadAsync(fullImageUrl, fileUri);
    
    if (downloadResult.status !== 200) {
      console.error(`Download failed with status: ${downloadResult.status}`);
      throw new Error('Failed to download file');
    }
    
    console.log(`Successfully downloaded image to: ${fileUri}`);
    return fileUri;
  } catch (error) {
    console.error('Error downloading image:', error);
    throw error;
  }
};

/**
 * Delete a processed image from the server
 * @param {string} filename - Name of the image to delete
 * @returns {Promise<Object>} - Result of the delete operation
 */
export const deleteProcessedImage = async (filename) => {
  try {
    console.log(`Deleting image: ${filename}`);
    
    const response = await fetchWithTimeout(`${API_URL}/processed/${filename}`, {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
      }
    }, 10000); // 10 second timeout
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error(`Failed to delete image: ${JSON.stringify(errorData)}`);
      throw new Error(errorData.error || 'Failed to delete image');
    }
    
    const data = await response.json();
    console.log(`Successfully deleted image: ${filename}`);
    
    return data;
  } catch (error) {
    console.error('Error deleting image:', error);
    throw error;
  }
}; 