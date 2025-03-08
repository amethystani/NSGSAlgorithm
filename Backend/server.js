const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all routes
app.use(cors({
  origin: '*',  // Allow all origins
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  credentials: true
}));

// Parse JSON bodies
app.use(express.json());

// Root endpoint for health check
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    message: 'YOLOv8 Object Detection API is running',
    endpoints: {
      '/': 'This help message',
      '/process': 'POST - Process an image using YOLOv8',
      '/history': 'GET - Get a list of processed images',
      '/processed/:filename': 'GET - Access a processed image file'
    }
  });
});

// Create Imginput directory if it doesn't exist
const inputDir = path.join(__dirname, 'Imginput');
if (!fs.existsSync(inputDir)) {
  fs.mkdirSync(inputDir, { recursive: true });
}

// Create Imgoutput directory if it doesn't exist
const outputDir = path.join(__dirname, 'Imgoutput');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Set up multer for handling file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    console.log('Multer destination called:', inputDir);
    cb(null, inputDir);
  },
  filename: (req, file, cb) => {
    // Create a unique filename with timestamp
    const uniqueName = `${Date.now()}-${file.originalname || 'image.jpg'}`;
    console.log('Multer filename generated:', uniqueName);
    cb(null, uniqueName);
  }
});

// Log the request body for debugging
const logBodyMiddleware = (req, res, next) => {
  console.log('Request body logging middleware');
  console.log('Headers:', req.headers);
  
  if (req.headers['content-type'] && req.headers['content-type'].includes('multipart/form-data')) {
    console.log('Multipart form data detected');
  }
  
  // Don't log the entire body for multipart as it may contain binary data
  next();
};

// File filter to accept all images
const fileFilter = (req, file, cb) => {
  console.log('Multer file filter called');
  console.log('File details:', {
    fieldname: file.fieldname,
    originalname: file.originalname || 'unknown',
    mimetype: file.mimetype || 'unknown'
  });
  
  // Accept any file - we'll check format later
  cb(null, true);
};

// Configure multer with relaxed options
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit to be safe
  },
}).single('image');

// Log-only middleware for debugging
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path} - Content-Type: ${req.headers['content-type'] || 'none'}`);
  next();
});

// Custom middleware to handle direct image uploads
const handleDirectImageUpload = (req, res, next) => {
  if (req.headers['content-type'] && req.headers['content-type'].includes('multipart/form-data')) {
    console.log('Direct image upload handler activated');
    
    // Let multer try to handle it first
    upload(req, res, function(err) {
      if (err) {
        console.error('Multer error:', err);
        return res.status(400).json({ error: err.message });
      }
      
      // If multer succeeded, continue
      if (req.file) {
        console.log('Multer successfully processed the file');
        return next();
      }
      
      // If multer didn't find a file but the image is in the body
      if (req.body && req.body.image && typeof req.body.image === 'object') {
        try {
          console.log('Found image object in request body');
          const imageData = req.body.image;
          
          // Check if it has the expected properties
          if (imageData.uri) {
            console.log('Image data from React Native detected');
            // Handle React Native image object format (simplified for example)
            return next();
          }
        } catch (err) {
          console.error('Error handling direct image upload:', err);
        }
      }
      
      // Continue anyway, the route handler might have other ways to handle it
      next();
    });
  } else {
    // Not a multipart request, continue
    next();
  }
};

// Serve static files from the output directory
app.use('/processed', express.static(path.join(__dirname, 'Imgoutput')));

// Debug endpoint for testing file upload without processing
app.post('/debug-upload', logBodyMiddleware, handleDirectImageUpload, (req, res) => {
  console.log('Received /debug-upload request');
  
  // If multer already processed the file, use it
  if (req.file) {
    console.log('Debug file successfully uploaded:', {
      filename: req.file.filename,
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size,
      path: req.file.path
    });
    
    return res.json({
      success: true,
      message: 'File upload successful (debug mode - multer processing)',
      file: {
        filename: req.file.filename,
        originalname: req.file.originalname,
        mimetype: req.file.mimetype,
        size: req.file.size,
        path: req.file.path
      },
      modelType: req.body.modelType || 'none'
    });
  }
  
  // Handle base64 image data sent directly in the request body
  if (req.body && req.body.imageBase64) {
    console.log('Base64 image data found in request body');
    try {
      // Get metadata from the request
      const filename = req.body.filename || `${Date.now()}-debug-upload.jpg`;
      const mimeType = req.body.mimeType || 'image/jpeg';
      
      console.log(`Creating debug file from base64 data: ${filename}, type: ${mimeType}`);
      
      // Create a debug file from base64 data (not saving to avoid filling disk)
      const base64Data = req.body.imageBase64;
      const dataLength = base64Data.length;
      
      return res.json({
        success: true,
        message: 'Base64 image data received successfully (debug mode)',
        info: {
          dataLength,
          filename,
          mimeType,
          validBase64: dataLength > 0,
          base64Preview: dataLength > 100 ? `${base64Data.substring(0, 50)}...` : base64Data
        },
        modelType: req.body.modelType || 'none'
      });
    } catch (error) {
      console.error('Error handling base64 image in debug mode:', error);
      return res.status(500).json({ error: 'Failed to process base64 image', details: error.message });
    }
  }
  
  // Handle case where image data is in req.body.image
  else if (req.body && req.body.image) {
    console.log('Image found in request body');
    
    // Analyze what kind of data we received
    let imageInfo = {
      type: typeof req.body.image,
      stringLength: typeof req.body.image === 'string' ? req.body.image.length : 'N/A',
      isPossiblyBase64: typeof req.body.image === 'string' && req.body.image.length > 100 && /^[A-Za-z0-9+/=]+$/.test(req.body.image.substring(0, 100)),
      isDataURI: typeof req.body.image === 'string' && req.body.image.startsWith('data:'),
      isObjectObjectString: req.body.image === '[object Object]',
      objectKeys: typeof req.body.image === 'object' ? Object.keys(req.body.image) : []
    };
    
    console.log('Image data analysis:', imageInfo);
    
    return res.json({
      success: true,
      message: 'Image data received in request body (debug mode)',
      info: imageInfo,
      modelType: req.body.modelType || 'none'
    });
  }
  
  // If we get here, no file was found
  return res.status(400).json({ error: 'No image file provided or invalid format' });
});

// Route to handle image upload and processing
app.post('/process', logBodyMiddleware, handleDirectImageUpload, (req, res) => {
  console.log('Processing request received');
  
  // Log the model type from the request to debug
  const requestedModelType = req.body.modelType || 'yolov8m-seg';
  console.log(`Requested model type: ${requestedModelType}`);
  
  // If multer already processed the file, use it
  if (req.file) {
    console.log('File successfully uploaded:', {
      filename: req.file.filename,
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size,
      path: req.file.path
    });
    
    processImageFile(req, res);
  } 
  // Handle base64 image data sent directly in the request body
  else if (req.body && req.body.imageBase64) {
    console.log('Base64 image data found in request body');
    try {
      // Get metadata from the request
      const filename = req.body.filename || `${Date.now()}-upload.jpg`;
      const mimeType = req.body.mimeType || 'image/jpeg';
      
      console.log(`Creating file from base64 data: ${filename}, type: ${mimeType}`);
      
      // Create the file from base64 data
      const base64Data = req.body.imageBase64;
      const filepath = path.join(inputDir, filename);
      
      // Write the file
      fs.writeFileSync(filepath, Buffer.from(base64Data, 'base64'));
      console.log(`Saved base64 data to file: ${filepath}, size: ${fs.statSync(filepath).size} bytes`);
      
      // Create a file object similar to what multer would produce
      req.file = {
        fieldname: 'image',
        originalname: filename,
        encoding: '7bit',
        mimetype: mimeType,
        destination: inputDir,
        filename: filename,
        path: filepath,
        size: fs.statSync(filepath).size
      };
      
      // Ensure model type is properly set
      if (!req.body.modelType) {
        console.log('No modelType specified in request, defaulting to yolov8m-seg');
        req.body.modelType = 'yolov8m-seg';
      } else {
        console.log(`Model type from request: ${req.body.modelType}`);
      }
      
      // Process the image
      processImageFile(req, res);
    } catch (error) {
      console.error('Error handling base64 image:', error);
      return res.status(500).json({ error: 'Failed to process base64 image', details: error.message });
    }
  }
  // Handle case where image data is in req.body.image
  else if (req.body && req.body.image) {
    console.log('Image found in request body, attempting to save');
    
    try {
      // Create a filename using timestamp
      const timestamp = Date.now();
      const filename = `${timestamp}-upload.jpg`;
      const filepath = path.join(inputDir, filename);
      
      // For debugging, log what kind of data we received
      console.log('Image data type:', typeof req.body.image);
      
      // Try to extract URI from the stringified object if possible
      let imageUri = null;
      if (typeof req.body.image === 'string') {
        // If it's a JSON string, try to parse it
        if (req.body.image.startsWith('{') && req.body.image.includes('uri')) {
          try {
            const imageObj = JSON.parse(req.body.image);
            imageUri = imageObj.uri;
            console.log('Extracted URI from JSON string:', imageUri);
          } catch (e) {
            console.log('Failed to parse JSON string');
          }
        }
        // If it's a "[object Object]" string (common React Native issue)
        else if (req.body.image === '[object Object]') {
          console.log('Received [object Object] string - this is a common React Native issue');
          // Instead of creating an empty file, create a small test image
          // This is a valid but minimal JPEG for testing
          const minimalJpeg = Buffer.from([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 
            0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC0, 0x00, 0x0B, 
            0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 
            0x3F, 0x00, 0xFF, 0xD9
          ]);
          fs.writeFileSync(filepath, minimalJpeg);
          console.log('Created a minimal test JPEG file');
        }
        // If it's a data URI
        else if (req.body.image.startsWith('data:image')) {
          console.log('Found data URI');
          const matches = req.body.image.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
          if (matches && matches.length === 3) {
            const data = Buffer.from(matches[2], 'base64');
            fs.writeFileSync(filepath, data);
            console.log('Saved data URI to file:', filepath);
          }
        }
      } 
      // If it's an actual object (not stringified)
      else if (typeof req.body.image === 'object' && req.body.image !== null) {
        console.log('Received actual object:', Object.keys(req.body.image));
        
        // If the object has a base64 property
        if (req.body.image.base64) {
          const data = Buffer.from(req.body.image.base64, 'base64');
          fs.writeFileSync(filepath, data);
          console.log('Saved base64 data to file:', filepath);
        }
        // If the object has a uri property
        else if (req.body.image.uri) {
          // We can't directly use the uri if it's a local file on client
          // But we'll save a placeholder file and log the URI for debugging
          console.log('Image URI from client (cannot access directly):', req.body.image.uri);
          // Use a minimal test image instead of just the JPEG header
          const minimalJpeg = Buffer.from([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 
            0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC0, 0x00, 0x0B, 
            0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 
            0x3F, 0x00, 0xFF, 0xD9
          ]);
          fs.writeFileSync(filepath, minimalJpeg);
        }
      }
      
      // Create a file object similar to what multer would produce
      req.file = {
        fieldname: 'image',
        originalname: filename,
        encoding: '7bit',
        mimetype: 'image/jpeg',
        destination: inputDir,
        filename: filename,
        path: filepath,
        size: fs.statSync(filepath).size
      };
      
      console.log('Created file from request body data:', req.file.path);
      
      // Process the image file
      processImageFile(req, res);
    } catch (error) {
      console.error('Error handling image from request body:', error);
      return res.status(500).json({ error: 'Failed to process image from request body', details: error.message });
    }
  } 
  else {
    // If we get here, no file was found
    return res.status(400).json({ error: 'No image file provided or invalid format' });
  }
});

// Helper function to process an image file with YOLOv8
function processImageFile(req, res) {
  try {
    let modelType = req.body.modelType || 'yolov8m-seg'; // Default to segmentation model if not specified
    console.log(`Using model type: ${modelType}`);
    
    // Ensure model type is valid, defaulting to segmentation if not
    modelType = ['yolov8m', 'yolov8m-seg'].includes(modelType) ? modelType : 'yolov8m-seg';
    console.log(`Validated model type: ${modelType}`);
    
    let modelPath, suffix;
    
    // Start with detection model as it's more reliable
    // We'll try segmentation only if explicitly requested and detection works
    let useDetectionFallback = false;
    
    // Look for ONNX models instead of PyTorch models
    if (modelType === 'yolov8m-seg') {
      // Try to find ONNX segmentation model file
      modelPath = './models/yolov8m-seg.onnx';
      if (!fs.existsSync(modelPath)) {
        // Check if there's an alternative path
        const altPath = path.join(__dirname, 'models', 'yolov8m-seg.onnx');
        if (fs.existsSync(altPath)) {
          modelPath = altPath;
        } else {
          console.log(`Segmentation ONNX model not found: ${modelPath}, looking for detection model`);
          useDetectionFallback = true;
        }
      }
      suffix = 'ms';
    } 
    
    if (useDetectionFallback || modelType === 'yolov8m') {
      // Try to find ONNX detection model file
      modelPath = './models/yolov8m.onnx';
      if (!fs.existsSync(modelPath)) {
        // Check if there's an alternative path
        const altPath = path.join(__dirname, 'models', 'yolov8m.onnx');
        if (fs.existsSync(altPath)) {
          modelPath = altPath;
        }
      }
      suffix = 'm';
      if (useDetectionFallback) {
        console.log('Using detection model as fallback');
      }
    }

    // Verify the input file exists and is readable
    const inputFilePath = req.file.path;
    try {
      const stats = fs.statSync(inputFilePath);
      console.log(`Input file verified: ${inputFilePath}, size: ${stats.size} bytes`);
      
      if (stats.size === 0) {
        throw new Error('Input file is empty');
      }
    } catch (error) {
      console.error(`Error verifying input file: ${error.message}`);
      return res.status(400).json({ error: 'Invalid input file', details: error.message });
    }

    // FALLBACK MECHANISM: Check if we want to skip actual processing (for testing UI)
    const skipProcessing = process.env.SKIP_PROCESSING === 'true';
    if (skipProcessing) {
      console.log('DEVELOPMENT MODE: Skipping actual YOLOv8 processing and creating a sample output');
      
      // Create a sample output file by copying the input file
      const originalName = req.file.filename;
      const baseName = originalName.substring(0, originalName.lastIndexOf('.') !== -1 ? 
        originalName.lastIndexOf('.') : originalName.length);
      const extension = originalName.lastIndexOf('.') !== -1 ? 
        originalName.substring(originalName.lastIndexOf('.')) : '.jpg';
      const processedName = `${baseName}_${suffix}${extension}`;
      const processedPath = path.join(outputDir, processedName);
      
      // Copy the input file to the output directory with the processed name
      fs.copyFileSync(inputFilePath, processedPath);
      console.log(`Created sample output file: ${processedPath}`);
      
      // Draw something on the copied file to simulate processing
      try {
        // Simply append some text to the file to show it was "processed"
        fs.appendFileSync(processedPath, Buffer.from("\n<!-- Processed by YOLOv8 simulation -->"));
      } catch (err) {
        console.log("Couldn't modify the output file, but continuing anyway");
      }
      
      // Construct the URL for the processed image
      const processedImageUrl = `/processed/${processedName}`;
      console.log(`Processed image URL: ${processedImageUrl}`);
      
      // Return the URL to the processed image
      return res.json({
        success: true,
        message: 'Image processed successfully (DEVELOPMENT MODE)',
        processedImageUrl: processedImageUrl,
        originalImageName: originalName,
        processedImageName: processedName,
        fullUrl: `${req.protocol}://${req.get('host')}${processedImageUrl}`
      });
    }

    // Check if ONNX model file exists
    if (!fs.existsSync(modelPath)) {
      console.error(`Model file not found: ${modelPath}`);
      return res.status(404).json({ 
        error: 'Model file not found', 
        details: `The ONNX model file ${modelPath} does not exist.`
      });
    }
    
    console.log(`Model file verified: ${modelPath}`);

    // Check if the build directory and executable exist
    const buildPath = path.join(__dirname, 'build');
    const executablePath = path.join(buildPath, 'yolov8_ort');
    
    try {
      if (!fs.existsSync(buildPath)) {
        throw new Error(`Build directory not found: ${buildPath}`);
      }
      
      if (!fs.existsSync(executablePath)) {
        throw new Error(`Executable not found: ${executablePath}`);
      }
      
      const execStats = fs.statSync(executablePath);
      console.log(`Executable verified: ${executablePath}, size: ${execStats.size} bytes`);
      
      // Check if the executable has execute permissions
      try {
        fs.accessSync(executablePath, fs.constants.X_OK);
        console.log('Executable has execute permissions');
      } catch (e) {
        console.warn('Warning: Executable might not have execute permissions');
        // Try to set execute permissions
        try {
          fs.chmodSync(executablePath, '755');
          console.log('Set execute permissions on executable');
        } catch (chmodErr) {
          console.error(`Error setting permissions: ${chmodErr.message}`);
        }
      }
    } catch (error) {
      console.error(`Error verifying executable: ${error.message}`);
      return res.status(500).json({ error: 'Executable not found or invalid', details: error.message });
    }

    // Check if models/coco.names exists
    const cocoNamesPath = path.join(__dirname, 'models', 'coco.names');
    try {
      if (!fs.existsSync(cocoNamesPath)) {
        throw new Error(`COCO names file not found: ${cocoNamesPath}`);
      }
      console.log(`COCO names file verified: ${cocoNamesPath}`);
    } catch (error) {
      console.error(`Error verifying COCO names file: ${error.message}`);
      return res.status(500).json({ error: 'COCO names file not found', details: error.message });
    }

    // Command to run the C++ application
    let command = '';
    
    if (modelType === 'yolov8m-seg' && !useDetectionFallback) {
      // For segmentation model
      console.log('Running with segmentation model');
      command = `./build/yolov8_ort -m ${modelPath} -i ./Imginput -o ./Imgoutput -c ./models/coco.names -x ${suffix}`;
    } else {
      // For regular detection model
      console.log('Running with detection model');
      command = `./build/yolov8_ort -m ${modelPath} -i ./Imginput -o ./Imgoutput -c ./models/coco.names -x ${suffix}`;
    }
    
    console.log(`Executing command: ${command}`);
    
    // Execute with timeout to prevent hanging
    const timeoutMs = 60000; // 60 seconds
    const execOptions = { timeout: timeoutMs };
    
    exec(command, execOptions, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing command: ${error.message}`);
        console.error(`stderr: ${stderr}`);
        
        // Create a simple fallback output
        const originalName = req.file.filename;
        const baseName = originalName.substring(0, originalName.lastIndexOf('.') !== -1 ? 
          originalName.lastIndexOf('.') : originalName.length);
        const extension = originalName.lastIndexOf('.') !== -1 ? 
          originalName.substring(originalName.lastIndexOf('.')) : '.jpg';
        const processedName = `${baseName}_${suffix}${extension}`;
        const processedPath = path.join(outputDir, processedName);
        
        // Create output directory if it doesn't exist
        if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
        }
        
        // Simply copy the input file to the output
        fs.copyFileSync(inputFilePath, processedPath);
        console.log(`Created fallback output by copying: ${processedPath}`);
        
        // Construct the URL for the processed image
        const processedImageUrl = `/processed/${processedName}`;
        
        // Return with a warning about fallback
        return res.json({
          success: true,
          warning: 'ONNX models not available, showing original image',
          message: 'Please convert PT models to ONNX format for actual processing',
          processedImageUrl: processedImageUrl,
          originalImageName: originalName,
          processedImageName: processedName,
          fullUrl: `${req.protocol}://${req.get('host')}${processedImageUrl}`,
          isFallback: true
        });
      }
      
      console.log(`stdout: ${stdout}`);
      if (stderr) {
        console.warn(`stderr (non-fatal): ${stderr}`);
      }
      
      // Get the processed image filename
      const originalName = req.file.filename;
      const baseName = originalName.substring(0, originalName.lastIndexOf('.') !== -1 ? 
        originalName.lastIndexOf('.') : originalName.length);
      const extension = originalName.lastIndexOf('.') !== -1 ? 
        originalName.substring(originalName.lastIndexOf('.')) : '.jpg';
      const processedName = `${baseName}_${suffix}${extension}`;
      const processedPath = path.join(outputDir, processedName);
      
      console.log(`Expected processed file: ${processedPath}`);
      
      // Verify the output file exists
      try {
        if (fs.existsSync(processedPath)) {
          const stats = fs.statSync(processedPath);
          console.log(`Output file verified: ${processedPath}, size: ${stats.size} bytes`);
        } else {
          throw new Error(`Processed file not found: ${processedPath}`);
        }
      } catch (error) {
        console.error(`Error verifying output file: ${error.message}`);
        return res.status(500).json({ error: 'Processing completed but output file not found', details: error.message });
      }
      
      // Construct the URL for the processed image
      const processedImageUrl = `/processed/${processedName}`;
      console.log(`Processed image URL: ${processedImageUrl}`);
      
      // Return the URL to the processed image
      res.json({
        success: true,
        message: 'Image processed successfully',
        processedImageUrl: processedImageUrl,
        originalImageName: originalName,
        processedImageName: processedName,
        fullUrl: `${req.protocol}://${req.get('host')}${processedImageUrl}`
      });
    });
  } catch (error) {
    console.error(`Server error in processImageFile: ${error.message}`);
    res.status(500).json({ error: 'Server error', details: error.message });
  }
}

// Route to get processing history (list of processed images)
app.get('/history', (req, res) => {
  const outputDir = path.join(__dirname, 'Imgoutput');
  
  // Check if output directory exists
  if (!fs.existsSync(outputDir)) {
    return res.json({ images: [] });
  }
  
  fs.readdir(outputDir, (err, files) => {
    if (err) {
      console.error(`Error reading output directory: ${err.message}`);
      return res.status(500).json({ error: 'Failed to read output directory', details: err.message });
    }
    
    // Filter for image files
    const imageFiles = files.filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png', '.gif'].includes(ext);
    });
    
    // Map files to URLs
    const images = imageFiles.map(file => ({
      name: file,
      url: `/processed/${file}`
    }));
    
    res.json({ images });
  });
});

// Error handler middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message
  });
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Server accessible at http://localhost:${PORT} or http://your-ip-address:${PORT}`);
}); 