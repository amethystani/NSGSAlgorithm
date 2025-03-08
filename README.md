# YOLOv8 Object Detection App

A mobile application that allows users to detect objects in images using the YOLOv8 model. The app consists of a React Native (Expo) frontend and a C++ backend with a Node.js API server.

## Project Structure

- **Frontend**: React Native (Expo) mobile application
- **Backend**: YOLOv8 C++ application for object detection with Node.js API server

## Prerequisites

- Node.js (v14+)
- npm or yarn
- OpenCV 4.2+
- ONNXRuntime 1.15+
- C++ compiler
- CUDA (optional, for GPU acceleration)

## Getting Started

### Backend Setup

1. Build the C++ application:

```bash
cd Backend
sh build.sh
```

2. Install the Node.js API server dependencies:

```bash
cd Backend
npm install
```

3. Start the API server:

```bash
cd Backend
npm start
```

The server will run on port 3000 by default.

### Frontend Setup

1. Install dependencies:

```bash
cd Frontend
npm install
```

2. Update the API URL in `Frontend/api/imageProcessingApi.js` if needed (if your backend server is running on a different address).

3. Start the Expo development server:

```bash
cd Frontend
npm run dev
```

## Features

- Take photos or select images from the gallery
- Process images with YOLOv8 object detection or segmentation models
- View processing history
- Download processed images to the device gallery

## API Endpoints

- `POST /process`: Upload and process an image with the YOLOv8 model
- `GET /history`: Get a list of processed images
- `GET /processed/:filename`: Get a specific processed image

## How It Works

1. The user takes a photo or selects an image from the gallery in the mobile app
2. The image is uploaded to the backend server
3. The server saves the image to the input directory
4. The server runs the YOLOv8 C++ application to process the image
5. The processed image is saved to the output directory
6. The server returns the URL of the processed image to the frontend
7. The frontend displays the processed image to the user

## License

[MIT License](LICENSE) 