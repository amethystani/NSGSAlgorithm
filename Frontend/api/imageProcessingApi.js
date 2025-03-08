import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';

// Determine the correct API URL based on platform
// Use the specific IP address for all devices when testing with Expo Go
export const getApiUrl = () => {
  // Use localhost for web
  if (Platform.OS === 'web') {
    return 'http://localhost:3000';
  }
  
  // For devices (iOS/Android), use the network IP
  // This is the actual IP address of your machine on the network
  const DEVICE_IP = '10.79.104.80';
  
  return `http://${DEVICE_IP}:3000`;
};

const API_URL = getApiUrl();
console.log(`Using API URL: ${API_URL}`);

// Helper function to create a fetch request with timeout
const fetchWithTimeout = async (url, options = {}, timeout = 30000) => {
  const controller = new AbortController();
  const signal = controller.signal;
  
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeout}ms`);
    }
    throw error;
  }
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
      console.log('Native platform detected, using FileSystem.readAsStringAsync');
      
      // For React Native, we need to get the actual image data as base64
      let imageBase64;
      try {
        imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
          encoding: FileSystem.EncodingType.Base64,
        });
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

/**
 * Process an image with YOLOv8
 * @param {string} imageUri - Local URI of the image to process
 * @param {string} modelType - Model type to use ('yolov8m' or 'yolov8m-seg')
 * @returns {Promise<Object>} - Result with processed image information
 */
export const processImage = async (imageUri, modelType = 'yolov8m') => {
  try {
    console.log(`Processing image with URI: ${imageUri} using model: ${modelType}`);
    
    // Validate model type
    if (!['yolov8m', 'yolov8m-seg'].includes(modelType)) {
      console.warn(`Invalid model type: ${modelType}, defaulting to yolov8m-seg`);
      modelType = 'yolov8m-seg';
    }
    
    // First test if the API is reachable
    const isApiConnected = await testApiConnection();
    if (!isApiConnected) {
      throw new Error(`Failed to connect to the API at ${API_URL}. Please check if the server is running.`);
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
        console.log(`Model type being sent: ${modelType}`);
        
        console.log(`Sending web fetch request to: ${API_URL}/process`);
        
        const result = await fetchWithTimeout(`${API_URL}/process`, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          }
        }, 60000).then(res => {
          if (!res.ok) {
            return res.text().then(text => {
              throw new Error(text || `Server responded with status ${res.status}`);
            });
          }
          return res.json();
        });
        
        console.log('Processing result:', result);
        return result;
      } catch (error) {
        console.error('Error in web image processing:', error);
        throw error;
      }
    } else {
      // Native platforms (iOS/Android)
      console.log('Native platform detected, using FileSystem.readAsStringAsync');
      
      // For React Native, we need to get the actual image data as base64
      let imageBase64;
      try {
        imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
          encoding: FileSystem.EncodingType.Base64,
        });
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
      
      // Add model type explicitly and with better logging
      console.log(`Setting model type in request to: ${modelType}`);
      formData.append('modelType', modelType);
      
      console.log(`Sending native request to: ${API_URL}/process with base64 data and model type: ${modelType}`);
      
      try {
        // Use fetch with the base64 data and increased timeout
        const response = await fetchWithTimeout(`${API_URL}/process`, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
          }
        }, 60000); // Increased timeout to 60 seconds for segmentation
        
        if (!response.ok) {
          const text = await response.text();
          console.error(`Server responded with ${response.status}: ${text}`);
          throw new Error(text || `Server responded with status ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Processing result:', result);
        
        // Add a timestamp to the URL to prevent caching
        if (result.fullUrl) {
          result.fullUrl = `${result.fullUrl}?t=${new Date().getTime()}`;
        }
        if (result.processedImageUrl) {
          result.processedImageUrl = `${result.processedImageUrl}?t=${new Date().getTime()}`;
        }
        
        return result;
      } catch (networkError) {
        console.error('Network error during processing:', networkError);
        throw new Error(`Network error: ${networkError.message}`);
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