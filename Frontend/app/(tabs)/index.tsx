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
import { ProcessedImageResult, DebugUploadResult } from '@/api/types';
import { triggerHaptic, triggerHapticNotification } from '@/utils/haptics';

const AnimatedBlurView = Animated.createAnimatedComponent(BlurView);

const BORDER_RADIUS = 12; // Standardized border radius
const CARD_SHADOW = {
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.1,
  shadowRadius: 4,
  elevation: 2
};

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
  const [processingTimer, setProcessingTimer] = useState(0);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const { theme, isDarkMode } = useTheme();

  // Helper function to create standardized card style with consistent shadows
  const getStandardCardStyle = useCallback((customStyle = {}) => {
    return [
      styles.standardCard,
      { 
        ...(isDarkMode ? {} : CARD_SHADOW),
        ...customStyle
      }
    ];
  }, [isDarkMode]);

  // Test API connection on component mount
  useEffect(() => {
    checkApiConnection();
  }, []);

  // Timer interval for processing time
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    
    if (processing) {
      // Reset timer when processing starts
      setProcessingTimer(0);
      interval = setInterval(() => {
        setProcessingTimer(prev => prev + 1);
      }, 1000);
    } else if (interval) {
      clearInterval(interval);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [processing]);

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
      
      try {
        // Use simpler options that should work across Expo versions
        let options = {
          quality: 0.8,
          allowsEditing: true,
          aspect: [4, 3] as [number, number],
        };
        
        // Platform-specific handling if needed
        if (Platform.OS === 'ios') {
          console.log('Using iOS camera options');
        } else if (Platform.OS === 'android') {
          console.log('Using Android camera options');
        }
        
        const result = await ImagePicker.launchCameraAsync(options);
        console.log('Camera result:', result);

        if (!result.canceled && result.assets && result.assets.length > 0) {
          console.log('Selected camera image:', result.assets[0].uri);
          setSelectedImage(result.assets[0].uri);
          setProcessedImage(null);
          setError(null); // Clear any previous errors
        }
      } catch (pickErr) {
        console.error('Specific error taking photo:', pickErr);
        
        // Try a more compatible approach on error
        console.log('Trying fallback camera approach');
        
        // Use the most basic options possible
        const basicResult = await ImagePicker.launchCameraAsync({
          quality: 0.8,
        });
        
        if (!basicResult.canceled && basicResult.assets && basicResult.assets.length > 0) {
          console.log('Selected camera image with fallback:', basicResult.assets[0].uri);
          setSelectedImage(basicResult.assets[0].uri);
          setProcessedImage(null);
          setError(null);
        } else {
          throw new Error('Failed with fallback camera approach too');
        }
      }
    } catch (err) {
      console.error('Error taking photo:', err);
      setError(err instanceof Error ? err.message : 'Failed to take photo');
      Alert.alert('Error', `Failed to take photo: ${err instanceof Error ? err.message : 'Unknown error'}`);
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
      
      try {
        // Use simpler options that should work across Expo versions
        let options = {
          quality: 0.8,
          allowsEditing: true,
          aspect: [4, 3] as [number, number],
        };
        
        // Platform-specific handling if needed
        if (Platform.OS === 'ios') {
          console.log('Using iOS options');
        } else if (Platform.OS === 'android') {
          console.log('Using Android options');
        }
        
        const result = await ImagePicker.launchImageLibraryAsync(options);
        console.log('Gallery result:', result);

        if (!result.canceled && result.assets && result.assets.length > 0) {
          console.log('Selected gallery image:', result.assets[0].uri);
          setSelectedImage(result.assets[0].uri);
          setProcessedImage(null);
          setError(null); // Clear any previous errors
        }
      } catch (pickErr) {
        console.error('Specific error selecting image:', pickErr);
        
        // Try a more compatible approach on error
        console.log('Trying fallback approach');
        
        // Use the most basic options possible
        const basicResult = await ImagePicker.launchImageLibraryAsync({
          quality: 0.8,
        });
        
        if (!basicResult.canceled && basicResult.assets && basicResult.assets.length > 0) {
          console.log('Selected gallery image with fallback:', basicResult.assets[0].uri);
          setSelectedImage(basicResult.assets[0].uri);
          setProcessedImage(null);
          setError(null);
        } else {
          throw new Error('Failed with fallback approach too');
        }
      }
    } catch (err) {
      console.error('Error selecting image:', err);
      setError(err instanceof Error ? err.message : 'Failed to select image');
      Alert.alert('Error', `Failed to select image: ${err instanceof Error ? err.message : 'Unknown error'}`);
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
    setProcessingTime(null); // Reset processing time
    
    const startTime = Date.now();
    
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
      
      // Additional code to calculate total processing time
      const endTime = Date.now();
      const totalProcessingTime = Math.round((endTime - startTime) / 1000);
      setProcessingTime(totalProcessingTime);
      
      // Add processing time to the result for history
      if (result) {
        // Check if it's a ProcessedImageResult before setting processingTime
        if ('processedImageUrl' in result) {
          (result as ProcessedImageResult).processingTime = totalProcessingTime;
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
            style={getStandardCardStyle({
              borderRadius: BORDER_RADIUS,
              backgroundColor: theme.card,
              padding: 12,
              flex: 0.88,
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'flex-start',
            })}
            onPress={checkApiConnection}
            disabled={isTestingConnection}
          >
            <View style={styles.connectionIconContainer}>
              {isTestingConnection ? (
                <ActivityIndicator size="small" color={theme.primary} />
              ) : (
                <View style={[
                  styles.connectionDot, 
                  { 
                    backgroundColor: apiConnected === true 
                      ? '#4CAF50' // green for connected
                      : apiConnected === false 
                        ? '#F44336' // red for disconnected
                        : '#9E9E9E' // gray for unknown
                  }
                ]} />
              )}
            </View>
            <Text style={[styles.connectionText, { color: theme.text }]}>
              {isTestingConnection 
                ? 'Testing...' 
                : apiConnected === true 
                  ? 'Connected - Animesh MacOS' 
                  : apiConnected === false 
                    ? 'Disconnected' 
                    : 'Check Connection'
              }
            </Text>
          </TouchableOpacity>
          
          {/* Debug Mode Toggle - Icon Only */}
          <TouchableOpacity
            style={getStandardCardStyle({
              backgroundColor: debugMode ? theme.primary : theme.card,
              width: 48,
              height: 48,
              padding: 0,
              marginLeft: 8,
            })}
            onPress={toggleDebugMode}
          >
            <Bug 
              size={20} 
              color={debugMode ? '#fff' : theme.text} 
              strokeWidth={2.5}
            />
          </TouchableOpacity>
        </View>
        
        {/* Error message */}
        {error && (
          <View style={[
            styles.errorContainer, 
            { borderRadius: BORDER_RADIUS }
          ]}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        
        {/* Model selection */}
        <View style={styles.modelSelection}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Select Model:</Text>
          <View style={styles.modelButtons}>
            <TouchableOpacity
              style={getStandardCardStyle({
                backgroundColor: selectedModel === 'yolov8m' ? theme.primary : theme.card,
                marginRight: 6,
                flex: 1,
              })}
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
              style={getStandardCardStyle({
                backgroundColor: selectedModel === 'yolov8m-seg' ? theme.primary : theme.card,
                marginLeft: 6,
                flex: 1,
              })}
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
        <View style={getStandardCardStyle({
          height: 250,
          marginBottom: 24,
          overflow: 'hidden',
          padding: 0,
        })}>
          {selectedImage ? (
            <Image source={{ uri: selectedImage }} style={styles.imagePreview} />
          ) : (
            <View style={[
              styles.noImagePlaceholder, 
              { 
                backgroundColor: theme.card,
                borderRadius: BORDER_RADIUS 
              }
            ]}>
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
            style={getStandardCardStyle({
              flex: 1,
              marginRight: 6,
              backgroundColor: theme.card,
            })}
            onPress={takePhoto}
          >
            <Camera size={24} color={theme.text} />
            <Text style={[styles.actionButtonText, { color: theme.text }]}>Camera</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={getStandardCardStyle({
              flex: 1,
              marginLeft: 6,
              backgroundColor: theme.card,
            })}
            onPress={selectImage}
          >
            <Upload size={24} color={theme.text} />
            <Text style={[styles.actionButtonText, { color: theme.text }]}>Gallery</Text>
          </TouchableOpacity>
        </View>
        
        {/* Process button */}
        <TouchableOpacity
          style={getStandardCardStyle({
            backgroundColor: theme.primary,
            marginBottom: 24,
            opacity: (!selectedImage || processing) ? 0.7 : 1
          })}
          onPress={() => {
            console.log('Process button clicked!');
            handleProcessImage();
          }}
          disabled={!selectedImage || processing}
        >
          {processing ? (
            <View style={styles.processingContainer}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.timerText}>{processingTimer}s</Text>
            </View>
          ) : (
            <Text style={styles.processButtonText}>
              Process Image
            </Text>
          )}
        </TouchableOpacity>
        
        {/* Processed image result */}
        {processedImage && (
          <View style={[
            styles.resultContainer, 
            { 
              borderRadius: BORDER_RADIUS,
              ...(isDarkMode ? {} : CARD_SHADOW)
            }
          ]}>
            <View style={styles.resultHeader}>
              <Text style={[styles.sectionTitle, { color: theme.text }]}>Result:</Text>
              {processingTime !== null && (
                <Text style={[styles.processingTimeText, { color: theme.textSecondary }]}>
                  Processed in {processingTime}s
                </Text>
              )}
            </View>
            <View style={[
              styles.processedImageWrapper,
              { 
                borderRadius: BORDER_RADIUS,
                ...(isDarkMode ? {} : CARD_SHADOW)
              }
            ]}>
              <Image 
                source={{ uri: processedImage.startsWith('http') 
                  ? processedImage 
                  : `${getApiUrl()}${processedImage}` 
                }} 
                style={styles.processedImage} 
                resizeMode="contain"
              />
            </View>
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
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: -0.5,
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
  },
  modelSelection: {
    marginBottom: 24,
  },
  modelButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modelButtonText: {
    fontWeight: '600',
    fontSize: 15,
  },
  imagePreviewContainer: {
    height: 250,
    marginBottom: 24,
    overflow: 'hidden',
    borderWidth: 0,
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
  },
  noImageText: {
    marginTop: 10,
    fontSize: 16,
  },
  actionButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  actionButton: {
    flex: 1,
    padding: 16,
    marginHorizontal: 6,
    justifyContent: 'center',
    alignItems: 'center',
  },
  actionButtonText: {
    marginTop: 8,
    fontWeight: '500',
    fontSize: 15,
  },
  processButton: {
    padding: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  processButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  resultContainer: {
    marginTop: 8,
    marginBottom: 32,
    padding: 16,
    backgroundColor: 'rgba(0,0,0,0.02)',
  },
  processedImageWrapper: {
    height: 350,
    width: '100%',
    overflow: 'hidden',
    borderWidth: 0,
    backgroundColor: '#f8f8f8',
  },
  processedImage: {
    width: '100%',
    height: '100%',
    backgroundColor: 'transparent',
  },
  apiConnectionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  connectionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    flex: 0.83,
  },
  connectionIconContainer: {
    width: 28,
    height: 28,
    justifyContent: 'center',
    alignItems: 'center',
  },
  connectionDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1,
    elevation: 2,
  },
  connectionText: {
    marginLeft: 12,
    fontSize: 15,
    fontWeight: '500',
    flexShrink: 1,
  },
  debugButton: {
    width: 48,
    height: 48,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 0,
  },
  errorContainer: {
    backgroundColor: '#ffeeee',
    padding: 16,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#ffcccc',
  },
  errorText: {
    color: '#cc0000',
    fontSize: 15,
  },
  processingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  timerText: {
    color: '#fff',
    marginLeft: 10,
    fontWeight: '600',
    fontSize: 15,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  processingTimeText: {
    fontSize: 15,
    fontStyle: 'italic',
  },
  standardCard: {
    padding: 16,
    borderRadius: BORDER_RADIUS,
    justifyContent: 'center',
    alignItems: 'center',
  },
});