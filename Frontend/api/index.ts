// Import functions from image processing API
import {
  getApiUrl,
  testApiConnection,
  debugUploadImage,
  processImage,
  getProcessedImages,
  downloadProcessedImage
} from './imageProcessingApi';

// Export types from types file
export type {
  ProcessedImage,
  ProcessedImageResult,
  DebugUploadResult
} from './types';

/**
 * Delete a processed image from the server
 * @param {string} filename - Name of the image to delete
 * @returns {Promise<any>} - Result of the delete operation
 */
export const deleteProcessedImage = async (filename: string): Promise<any> => {
  try {
    console.log(`Deleting image: ${filename}`);
    const apiUrl = getApiUrl();
    
    // Use the new API endpoint for deletion
    const response = await fetch(`${apiUrl}/api/deleteImage/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
      }
    });
    
    if (!response.ok) {
      // Check if the response is JSON
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const errorData = await response.json();
        console.error(`Failed to delete image: ${JSON.stringify(errorData)}`);
        throw new Error(errorData.error || 'Failed to delete image');
      } else {
        // Handle non-JSON responses
        const text = await response.text();
        console.error(`Server returned non-JSON response: ${text}`);
        throw new Error(`Server error (${response.status}): Failed to delete image`);
      }
    }
    
    // Try to parse as JSON, but handle if it's not
    try {
      const data = await response.json();
      console.log(`Successfully deleted image: ${filename}`);
      return data;
    } catch (parseError) {
      // If the response isn't valid JSON but the status was OK, consider it a success
      console.log(`Successfully deleted image (with non-JSON response): ${filename}`);
      return { success: true };
    }
  } catch (error) {
    console.error('Error deleting image:', error);
    throw error;
  }
};

// Export functions
export {
  getApiUrl,
  testApiConnection,
  debugUploadImage,
  processImage,
  getProcessedImages,
  downloadProcessedImage
}; 