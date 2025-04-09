/**
 * ProcessedImage type definition used in both history and processing results
 */
export interface ProcessedImage {
  name: string;
  url: string;
  processingTime?: number;
  fileSize?: number;
  fileSizeFormatted?: string;
  width?: number;
  height?: number;
  modelType?: string;
  modelTypeName?: string;
  processedAt?: string;
  processedDate?: string;
  processedTime?: string;
  isFallback?: boolean;
  stackId?: string;
  isStackImage?: boolean;
}

/**
 * Result from processing an image with YOLOv8
 */
export interface ProcessedImageResult {
  success?: boolean;
  message?: string;
  warning?: string;
  processedImageUrl?: string;
  originalImageName?: string;
  processedImageName?: string;
  fullUrl?: string;
  isFallback?: boolean;
  processingTime?: number;
  usedNSGS?: boolean;
  stackId?: string;
  isStackImage?: boolean;
  nsgsStats?: {
    graphNodes: number;
    processedSpikes: number;
    queueSize: number;
    adaptationMultiplier: number;
    processingTime: number;
    status: string;
  };
}

/**
 * Result from uploading an image for debugging
 */
export interface DebugUploadResult {
  file?: {
    originalname?: string;
    size?: number;
  };
} 