import { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Platform,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, Upload, Image as ImageIcon, Wifi, Bug } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeIn,
  FadeOut,
  SlideInDown,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
  Easing,
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';
import { processImage, testApiConnection, debugUploadImage, getApiUrl } from '@/api';
import { triggerHaptic, triggerHapticNotification } from '@/utils/haptics';

const AnimatedBlurView = Animated.createAnimatedComponent(BlurView);

export default function DetectScreen() {
  const [selectedModel, setSelectedModel] = useState('yolov8m-seg');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isResultVisible, setIsResultVisible] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null); // null = unknown, true/false for connected status
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const { theme, isDarkMode } = useTheme();

  // Test API connection on component mount
  useEffect(() => {
    checkApiConnection();
  }, []);

  // Function to check API connection
  const checkApiConnection = useCallback(async () => {
    setIsTestingConnection(true);
    setError(null);
    try {
      const connected = await testApiConnection();
      setApiConnected(connected);
      if (!connected) {
        setError('Cannot connect to the API server. Please check if the server is running.');
      }
    } catch (err) {
      console.error('Error testing API connection:', err);
      setApiConnected(false);
      setError('Cannot connect to the API server. Please check if the server is running.');
    } finally {
      setIsTestingConnection(false);
    }
  }, []);

  // Function to trigger haptic feedback on model selection
  const handleModelSelect = useCallback((model: string) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedModel(model);
  }, []);

  // Function to take a photo with camera
  const takePhoto = useCallback(async () => {
    try {
      // Request camera permissions first
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      
      if (status !== 'granted') {
        setError('Camera permission is required to take photos');
        Alert.alert('Permission Required', 'Camera permission is needed to take photos.');
        return;
      }
      
      triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
      
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8, // Slightly reduce quality for better performance
        aspect: [4, 3],
        exif: false, // Don't need EXIF data
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        console.log('Selected camera image:', result.assets[0].uri);
        setSelectedImage(result.assets[0].uri);
        setProcessedImage(null);
        setError(null); // Clear any previous errors
      }
    } catch (err) {
      console.error('Error taking photo:', err);
      setError(err instanceof Error ? err.message : 'Failed to take photo');
      Alert.alert('Error', 'Failed to take photo. Please try again.');
    }
  }, []);

  // Function to select image from gallery
  const selectImage = useCallback(async () => {
    try {
      // Request media library permissions first
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (status !== 'granted') {
        setError('Media library permission is required to select images');
        Alert.alert('Permission Required', 'Media library permission is needed to select images.');
        return;
      }
      
      triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
      
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8, // Slightly reduce quality for better performance
        aspect: [4, 3],
        exif: false, // Don't need EXIF data
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        console.log('Selected gallery image:', result.assets[0].uri);
        setSelectedImage(result.assets[0].uri);
        setProcessedImage(null);
        setError(null); // Clear any previous errors
      }
    } catch (err) {
      console.error('Error selecting image:', err);
      setError(err instanceof Error ? err.message : 'Failed to select image');
      Alert.alert('Error', 'Failed to select image. Please try again.');
    }
  }, []);

  // Function to process the selected image
  const handleProcessImage = useCallback(async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select or take a photo first');
      return;
    }

    setProcessing(true);
    setError(null);
    setProcessedImage(null); // Clear any previous processed image
    
    try {
      // Log the image info before processing
      console.log(`Processing image: ${selectedImage} with model: ${selectedModel}`);
      
      // Log the API URL being used
      const apiUrl = getApiUrl();
      console.log(`API URL: ${apiUrl}`);
      
      // Trigger success haptic when processing starts
      triggerHapticNotification(Haptics.NotificationFeedbackType.Success);
      
      let result;
      
      if (debugMode) {
        // Use debug upload mode (just uploads the file without processing)
        console.log('Using debug mode - uploading without processing');
        
        try {
          // First test API connection
          const isConnected = await testApiConnection();
          console.log('API connection test result:', isConnected);
          
          if (!isConnected) {
            throw new Error(`Cannot connect to API at ${apiUrl}. Check that the server is running.`);
          }
          
          result = await debugUploadImage(selectedImage);
          console.log('Debug upload result:', result);
          
          Alert.alert('Debug Upload Success', 
            `File uploaded successfully without processing:\n${result.file?.originalname}\nSize: ${result.file?.size} bytes`
          );
        } catch (debugError: any) {
          console.error('Debug mode error:', debugError);
          throw new Error(`Debug upload failed: ${debugError.message}`);
        }
      } else {
        // First test API connection with explicit error handling
        try {
          console.log('Testing API connection...');
          const isConnected = await testApiConnection();
          console.log('API connection test result:', isConnected);
          
          if (!isConnected) {
            throw new Error(`Cannot connect to API at ${apiUrl}. Check that the server is running.`);
          }
        } catch (connectionError: any) {
          console.error('Connection test error:', connectionError);
          throw new Error(`API connection failed: ${connectionError.message}`);
        }
        
        // Normal processing mode
        try {
          console.log(`Processing with model: ${selectedModel}`);
          result = await processImage(selectedImage, selectedModel);
          console.log('Processing result:', result);
        } catch (processingError: any) {
          console.error('Image processing error:', processingError);
          throw new Error(`Processing failed: ${processingError.message}`);
        }
        
        // Force refresh the image cache by adding a timestamp parameter
        const timestamp = new Date().getTime();
        
        // Check for fullUrl in the response first
        if (result && result.fullUrl) {
          console.log(`Using full URL from server: ${result.fullUrl}`);
          // Always add a cache-busting parameter
          const cachebustedUrl = result.fullUrl.includes('?') 
            ? `${result.fullUrl}&t=${timestamp}` 
            : `${result.fullUrl}?t=${timestamp}`;
          setProcessedImage(cachebustedUrl);
        }
        // If no fullUrl, use processedImageUrl with API URL
        else if (result && result.processedImageUrl) {
          const fullUrl = `${apiUrl}${result.processedImageUrl}`;
          console.log(`Constructed URL from processedImageUrl: ${fullUrl}`);
          // Always add a cache-busting parameter
          const cachebustedUrl = fullUrl.includes('?') 
            ? `${fullUrl}&t=${timestamp}` 
            : `${fullUrl}?t=${timestamp}`;
          setProcessedImage(cachebustedUrl);
        } else {
          console.error('No image URL in response:', result);
          throw new Error('No processed image URL received from server');
        }
        
        // Show success message (or warning for fallback)
        if (result.isFallback) {
          Alert.alert(
            'Processing Warning', 
            'The segmentation model failed to process this image, so we\'re showing the original image. Try the object detection model instead.'
          );
        } else if (result.warning && result.warning.includes('fallback detection')) {
          Alert.alert(
            'Segmentation Notice', 
            'The segmentation model had issues, so we used object detection instead. The results are shown with bounding boxes rather than full segmentation masks.'
          );
        } else if (result.warning) {
          Alert.alert(
            'Processing Notice', 
            `The image was processed with a warning: ${result.warning}`
          );
        } else if (result.message && result.message.includes('best match')) {
          Alert.alert(
            'Processing Success',
            'The image was processed successfully, but we had to use the best matching output file.'
          );
        } else {
          Alert.alert('Success', 'Image processed successfully!');
        }
      }
      
      // Trigger success haptic when processing is complete
      triggerHapticNotification(Haptics.NotificationFeedbackType.Success);
    } catch (err: any) {
      console.error('Error processing image:', err);
      const errorMessage = err?.message || 'Failed to process image. Please try debug mode to troubleshoot.';
      setError(errorMessage);
      
      // Show error dialog with more details
      Alert.alert(
        'Processing Error', 
        `Error: ${errorMessage}\n\nTry enabling Debug mode to test the connection.`
      );
      
      // Trigger error haptic
      triggerHapticNotification(Haptics.NotificationFeedbackType.Error);
    } finally {
      setProcessing(false);
    }
  }, [selectedImage, selectedModel, debugMode]);

  // Toggle debug mode
  const toggleDebugMode = useCallback(() => {
    setDebugMode(!debugMode);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
  }, [debugMode]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: theme.background }}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={[styles.title, { color: theme.text }]}>Object Detection</Text>
        
        {/* API Connection Status */}
        <View style={styles.apiConnectionContainer}>
          <TouchableOpacity
            style={[styles.connectionButton, { backgroundColor: theme.card }]}
            onPress={checkApiConnection}
            disabled={isTestingConnection}
          >
            {isTestingConnection ? (
              <ActivityIndicator size="small" color={theme.primary} />
            ) : (
              <Wifi 
                size={20} 
                color={apiConnected === true 
                  ? '#4CAF50' // green for connected
                  : apiConnected === false 
                    ? '#F44336' // red for disconnected
                    : theme.textSecondary // default color for unknown
                } 
              />
            )}
            <Text style={[styles.connectionText, { color: theme.text }]}>
              {isTestingConnection 
                ? 'Testing...' 
                : apiConnected === true 
                  ? 'Connected' 
                  : apiConnected === false 
                    ? 'Disconnected' 
                    : 'Check Connection'
              }
            </Text>
          </TouchableOpacity>
          
          {/* Debug Mode Toggle */}
          <TouchableOpacity
            style={[
              styles.debugButton, 
              { 
                backgroundColor: debugMode ? theme.primary : theme.card,
                borderColor: theme.border
              }
            ]}
            onPress={toggleDebugMode}
          >
            <Bug size={20} color={debugMode ? '#fff' : theme.text} />
            <Text style={[
              styles.debugText,
              { color: debugMode ? '#fff' : theme.text }
            ]}>
              Debug
            </Text>
          </TouchableOpacity>
        </View>
        
        {/* Error message */}
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        
        {/* Model selection */}
        <View style={styles.modelSelection}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Select Model:</Text>
          <View style={styles.modelButtons}>
            <TouchableOpacity
              style={[
                styles.modelButton,
                selectedModel === 'yolov8m' && styles.selectedModelButton,
                { backgroundColor: selectedModel === 'yolov8m' ? theme.primary : theme.card }
              ]}
              onPress={() => handleModelSelect('yolov8m')}
            >
              <Text style={[
                styles.modelButtonText, 
                { color: selectedModel === 'yolov8m' ? '#fff' : theme.text }
              ]}>
                Object Detection
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.modelButton,
                selectedModel === 'yolov8m-seg' && styles.selectedModelButton,
                { backgroundColor: selectedModel === 'yolov8m-seg' ? theme.primary : theme.card }
              ]}
              onPress={() => handleModelSelect('yolov8m-seg')}
            >
              <Text style={[
                styles.modelButtonText, 
                { color: selectedModel === 'yolov8m-seg' ? '#fff' : theme.text }
              ]}>
                Segmentation
              </Text>
            </TouchableOpacity>
          </View>
        </View>
        
        {/* Image preview */}
        <View style={styles.imagePreviewContainer}>
          {selectedImage ? (
            <Image source={{ uri: selectedImage }} style={styles.imagePreview} />
          ) : (
            <View style={[styles.noImagePlaceholder, { backgroundColor: theme.card }]}>
              <ImageIcon size={50} color={theme.textSecondary} />
              <Text style={[styles.noImageText, { color: theme.textSecondary }]}>
                No image selected
              </Text>
            </View>
          )}
        </View>
        
        {/* Action buttons */}
        <View style={styles.actionButtonsContainer}>
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: theme.card }]}
            onPress={takePhoto}
          >
            <Camera size={24} color={theme.text} />
            <Text style={[styles.actionButtonText, { color: theme.text }]}>Camera</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: theme.card }]}
            onPress={selectImage}
          >
            <Upload size={24} color={theme.text} />
            <Text style={[styles.actionButtonText, { color: theme.text }]}>Gallery</Text>
          </TouchableOpacity>
        </View>
        
        {/* Process button */}
        <TouchableOpacity
          style={[
            styles.processButton,
            { backgroundColor: theme.primary },
            (!selectedImage || processing) && { opacity: 0.7 }
          ]}
          onPress={() => {
            console.log('Process button clicked!');
            handleProcessImage();
          }}
          disabled={!selectedImage || processing}
        >
          {processing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.processButtonText}>
              Process Image
            </Text>
          )}
        </TouchableOpacity>
        
        {/* Processed image result */}
        {processedImage && (
          <View style={styles.resultContainer}>
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Result:</Text>
            <Image 
              source={{ uri: processedImage.startsWith('http') 
                ? processedImage 
                : `${getApiUrl()}${processedImage}` 
              }} 
              style={styles.processedImage} 
              resizeMode="contain"
            />
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
  },
  modelSelection: {
    marginBottom: 20,
  },
  modelButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modelButton: {
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 5,
    alignItems: 'center',
  },
  selectedModelButton: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  modelButtonText: {
    fontWeight: '600',
  },
  imagePreviewContainer: {
    height: 250,
    marginBottom: 20,
    borderRadius: 12,
    overflow: 'hidden',
  },
  imagePreview: {
    width: '100%',
    height: '100%',
  },
  noImagePlaceholder: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 12,
  },
  noImageText: {
    marginTop: 10,
  },
  actionButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    width: '45%',
  },
  actionButtonText: {
    marginLeft: 8,
    fontWeight: '600',
  },
  processButton: {
    paddingVertical: 15,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  processButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  errorContainer: {
    marginBottom: 20,
  },
  errorText: {
    color: '#F44336', // Red error color
    textAlign: 'center',
    fontWeight: '600',
  },
  resultContainer: {
    marginTop: 10,
  },
  processedImage: {
    width: '100%',
    height: 250,
    borderRadius: 12,
  },
  apiConnectionContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  connectionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  connectionText: {
    marginLeft: 10,
    fontWeight: '600',
  },
  debugButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  debugText: {
    marginLeft: 10,
    fontWeight: '600',
  },
});