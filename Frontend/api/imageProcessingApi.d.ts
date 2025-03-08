// Type definitions for imageProcessingApi
export interface ProcessedImageResult {
  success: boolean;
  message: string;
  processedImageUrl: string;
  originalImageName: string;
  processedImageName: string;
  fullUrl?: string; // Optional full URL including hostname/port
  isFallback?: boolean; // Flag indicating if this is a fallback image
  details?: string; // Additional details, especially for errors
  error?: string; // Error message if success is false
  warning?: string; // Warning message for partial success cases
}

export interface ProcessedImage {
  name: string;
  url: string;
}

export interface DebugUploadResult {
  success: boolean;
  message: string;
  file?: {
    filename: string;
    originalname: string;
    mimetype: string;
    size: number;
    path: string;
  };
  modelType?: string;
  rawResponse?: string;
}

export function getApiUrl(): string;
export function testApiConnection(): Promise<boolean>;
export function debugUploadImage(imageUri: string): Promise<DebugUploadResult>;
export function processImage(imageUri: string, modelType?: string): Promise<ProcessedImageResult>;
export function getProcessedImages(): Promise<ProcessedImage[]>;
export function downloadProcessedImage(imageUrl: string, filename: string): Promise<string>; 