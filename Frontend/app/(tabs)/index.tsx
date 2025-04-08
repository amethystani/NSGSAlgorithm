import React, { useState, useCallback, useEffect, useRef } from 'react';
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
  Modal,
  FlatList,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, Upload, Image as ImageIcon, Wifi, Bug, BrainCircuit, Scan, Layers, Zap, Network, ChevronDown, LayoutGrid, CopyCheck } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeIn,
  FadeOut,
  SlideInDown,
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';
import { processImage, testApiConnection, debugUploadImage, getApiUrl } from '@/api';
import { ProcessedImageResult, DebugUploadResult } from '@/api/types';
import { triggerHaptic, triggerHapticNotification } from '@/utils/haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';

const AnimatedBlurView = Animated.createAnimatedComponent(BlurView);

const BORDER_RADIUS = 12; // Standardized border radius
const CARD_SHADOW = {
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.1,
  shadowRadius: 4,
  elevation: 2
};

// Define types for model and image mode options
interface ModelOption {
  id: string;
  label: string;
  value: string;
  icon: any; // Component type
  useNSGS: boolean;
  color?: string;
}

interface ImageModeOption {
  id: string;
  label: string;
  value: string;
  icon: any; // Component type
}

// Base type for option
type DropdownOption = ModelOption | ImageModeOption;

// Type guard to check if an option is a ModelOption
function isModelOption(option: DropdownOption): option is ModelOption {
  return 'useNSGS' in option;
}

// Define model options
const modelOptions: ModelOption[] = [
  { id: 'detection', label: 'Detection', value: 'yolov8m', icon: Scan, useNSGS: false },
  { id: 'segmentation', label: 'Segmentation', value: 'yolov8m-seg', icon: Layers, useNSGS: false },
  { id: 'nsgs', label: 'NSGS', value: 'yolov8m-seg', icon: BrainCircuit, useNSGS: true, color: '#6A35D9' },
];

// Define image mode options
const imageModeOptions: ImageModeOption[] = [
  { id: 'single', label: 'Single Image', value: 'single', icon: ImageIcon },
  { id: 'stack', label: 'Image Stack', value: 'stack', icon: LayoutGrid },
];

export default function DetectScreen() {
  const [selectedModel, setSelectedModel] = useState('yolov8m-seg');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isResultVisible, setIsResultVisible] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null); // null = unknown, true/false for connected status
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [processingTimer, setProcessingTimer] = useState(0);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [useNSGS, setUseNSGS] = useState(false); // Direct NSGS control
  const [result, setResult] = useState<ProcessedImageResult | null>(null); // Store the API response
  const { theme, isDarkMode } = useTheme();
  
  // Dropdown states
  const [modelDropdownVisible, setModelDropdownVisible] = useState(false);
  const [imageModeDropdownVisible, setImageModeDropdownVisible] = useState(false);
  const [selectedImageMode, setSelectedImageMode] = useState(imageModeOptions[0]);
  const [selectedModelOption, setSelectedModelOption] = useState(modelOptions[1]); // Default to segmentation

  // Load NSGS setting from AsyncStorage
  useEffect(() => {
    const loadNsgsPreference = async () => {
      try {
        const storedValue = await AsyncStorage.getItem('useNSGS');
        console.log(`Loaded NSGS preference from AsyncStorage: ${storedValue}`);
        if (storedValue !== null) {
          const parsedValue = JSON.parse(storedValue);
          console.log(`Parsed NSGS value: ${parsedValue} (${typeof parsedValue})`);
          setUseNSGS(parsedValue);
        }
      } catch (e) {
        console.error('Failed to load NSGS preference:', e);
      }
    };
    
    loadNsgsPreference();
  }, []);

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
        setProcessingTimer((prev: number) => prev + 1);
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
  const handleModelSelect = useCallback((model: string, useNSGSFlag?: boolean) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedModel(model);
    // If useNSGSFlag is explicitly provided, set it
    if (typeof useNSGSFlag === 'boolean') {
      setUseNSGS(useNSGSFlag);
      
      // Save NSGS preference to AsyncStorage
      try {
        AsyncStorage.setItem('useNSGS', JSON.stringify(useNSGSFlag));
        console.log(`Saved NSGS preference to AsyncStorage: ${useNSGSFlag}`);
      } catch (e) {
        console.error('Failed to save NSGS preference:', e);
      }
    } else {
      // Default to false for other models
      setUseNSGS(false);
      
      // Save NSGS preference to AsyncStorage when turning it off
      try {
        AsyncStorage.setItem('useNSGS', JSON.stringify(false));
        console.log('Saved NSGS preference to AsyncStorage: false');
      } catch (e) {
        console.error('Failed to save NSGS preference:', e);
      }
    }
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
          setSelectedImages([]);
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
          setSelectedImages([]);
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

  // Function to handle model option selection from dropdown
  const handleModelOptionSelect = (option: ModelOption) => {
    setSelectedModelOption(option);
    handleModelSelect(option.value, option.useNSGS);
    setModelDropdownVisible(false);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
  };

  // Function to handle image mode selection from dropdown
  const handleImageModeSelect = (option: ImageModeOption) => {
    setSelectedImageMode(option);
    setImageModeDropdownVisible(false);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    
    // Reset images if switching modes
    if (option.value === 'single') {
      setSelectedImages([]);
      if (selectedImages.length > 0) {
        setSelectedImage(selectedImages[0]);
      }
    } else {
      // If switching to stack and we have a single image, add it to the stack
      if (selectedImage && selectedImages.length === 0) {
        setSelectedImages([selectedImage]);
      }
    }
  };

  // Function to select multiple images for stack mode
  const selectMultipleImages = async () => {
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
        let options = {
          quality: 0.8,
          allowsMultipleSelection: true,
          selectionLimit: 10, // Allow up to 10 images
        };
        
        const result = await ImagePicker.launchImageLibraryAsync(options);
        console.log('Gallery multiple selection result:', result);

        if (!result.canceled && result.assets && result.assets.length > 0) {
          const uris = result.assets.map(asset => asset.uri);
          setSelectedImages(uris);
          if (uris.length > 0) {
            setSelectedImage(uris[0]); // Set the first image as preview
          }
          setProcessedImage(null);
          setError(null);
        }
      } catch (err) {
        console.error('Error selecting multiple images:', err);
        setError(err instanceof Error ? err.message : 'Failed to select images');
        Alert.alert('Error', `Failed to select images: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error with multi-select:', err);
      setError(err instanceof Error ? err.message : 'Failed to initialize multi-select');
    }
  };

  // Select image - updated to handle single/stack modes
  const selectImage = useCallback(async () => {
    if (selectedImageMode.value === 'stack') {
      await selectMultipleImages();
      return;
    }
    
    // Original single image selection code
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
          setSelectedImages([result.assets[0].uri]); // Also update selected images array
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
          setSelectedImages([basicResult.assets[0].uri]); // Also update selected images array
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
  }, [selectedImageMode]);

  // Function to process the selected image
  const handleProcessImage = useCallback(async () => {
    if (!selectedImage && selectedImages.length === 0) {
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
      console.log(`Processing image: ${selectedImage || selectedImages.join(', ')} with model: ${selectedModel}`);
      
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
          
          result = await debugUploadImage(selectedImage || selectedImages[0]);
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
          console.log(`NSGS mode: ${useNSGS ? 'Enabled ✓' : 'Disabled ✗'}`);
          result = await processImage(selectedImage || selectedImages[0], selectedModel, useNSGS);
          console.log('Processing result:', result);
          
          // Store the result in state
          setResult(result);
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
  }, [selectedImage, selectedImages, selectedModel, debugMode, useNSGS]);

  // Toggle debug mode
  const toggleDebugMode = useCallback(() => {
    setDebugMode(!debugMode);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
  }, [debugMode]);

  // Render dropdown for model selection
  const renderModelDropdown = (
    visible: boolean, 
    setVisible: React.Dispatch<React.SetStateAction<boolean>>, 
    options: ModelOption[], 
    selectedOption: ModelOption, 
    onSelect: (option: ModelOption) => void
  ) => {
    const backgroundColor = selectedOption.color || theme.card;
    
    return (
      <>
        <TouchableOpacity
          style={[
            styles.dropdownButton,
            { backgroundColor: backgroundColor }
          ]}
          onPress={() => setVisible(true)}
        >
          <View style={styles.dropdownSelectedItem}>
            {selectedOption.icon && (
              <selectedOption.icon 
                size={20} 
                color={selectedOption.color ? '#fff' : theme.text} 
                style={styles.dropdownIcon} 
              />
            )}
            <Text style={[
              styles.dropdownSelectedText, 
              { color: selectedOption.color ? '#fff' : theme.text }
            ]}>
              {selectedOption.label}
            </Text>
          </View>
          <ChevronDown size={16} color={selectedOption.color ? '#fff' : theme.text} />
        </TouchableOpacity>

        <Modal
          visible={visible}
          transparent={true}
          animationType="fade"
          onRequestClose={() => setVisible(false)}
        >
          <TouchableOpacity
            style={styles.dropdownOverlay}
            activeOpacity={1}
            onPress={() => setVisible(false)}
          >
            <Animated.View 
              entering={FadeIn.duration(150)}
              exiting={FadeOut.duration(100)}
              style={styles.dropdownAnimationContainer}
            >
              <BlurView 
                intensity={isDarkMode ? 15 : 40}
                tint={isDarkMode ? "dark" : "light"}
                style={styles.dropdownMenuBlur}
              >
                <View style={styles.dropdownHeader}>
                  <Text style={[styles.dropdownTitle, { color: theme.text }]}>
                    Select Model
                  </Text>
                </View>
                
                <View style={styles.dropdownDivider} />
                
                <FlatList
                  data={options}
                  keyExtractor={(item) => item.id}
                  showsVerticalScrollIndicator={false}
                  bounces={true}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      style={[
                        styles.dropdownItem,
                        selectedOption.id === item.id ? 
                          (item.color ? { backgroundColor: `${item.color}20` } : { backgroundColor: 'rgba(0, 122, 255, 0.08)' }) : {}
                      ]}
                      onPress={() => onSelect(item)}
                    >
                      <View style={styles.dropdownItemContent}>
                        <View style={[
                          styles.dropdownItemIconContainer,
                          { backgroundColor: selectedOption.id === item.id ? (item.color || 'rgba(0, 122, 255, 0.8)') : 'rgba(140, 140, 140, 0.08)' }
                        ]}>
                          {item.icon && (
                            <item.icon 
                              size={16} 
                              color={selectedOption.id === item.id ? '#fff' : theme.text} 
                              strokeWidth={2}
                            />
                          )}
                        </View>
                        <Text style={[
                          styles.dropdownItemText,
                          { 
                            color: theme.text,
                            fontWeight: selectedOption.id === item.id ? '500' : '400'
                          }
                        ]}>
                          {item.label}
                        </Text>
                      </View>
                      {selectedOption.id === item.id && (
                        <CopyCheck size={16} color={item.color || 'rgba(0, 122, 255, 0.8)'} strokeWidth={2} />
                      )}
                    </TouchableOpacity>
                  )}
                  style={styles.dropdownList}
                />
                
                <TouchableOpacity 
                  style={styles.dropdownCancelButton}
                  onPress={() => setVisible(false)}
                >
                  <Text style={styles.dropdownCancelText}>Cancel</Text>
                </TouchableOpacity>
              </BlurView>
            </Animated.View>
          </TouchableOpacity>
        </Modal>
      </>
    );
  };

  // Render dropdown for image mode selection
  const renderImageModeDropdown = (
    visible: boolean, 
    setVisible: React.Dispatch<React.SetStateAction<boolean>>, 
    options: ImageModeOption[], 
    selectedOption: ImageModeOption, 
    onSelect: (option: ImageModeOption) => void
  ) => {
    return (
      <>
        <TouchableOpacity
          style={[
            styles.dropdownButton,
            { backgroundColor: theme.card }
          ]}
          onPress={() => setVisible(true)}
        >
          <View style={styles.dropdownSelectedItem}>
            {selectedOption.icon && (
              <selectedOption.icon 
                size={20} 
                color={theme.text} 
                style={styles.dropdownIcon} 
              />
            )}
            <Text style={[
              styles.dropdownSelectedText, 
              { color: theme.text }
            ]}>
              {selectedOption.label}
            </Text>
          </View>
          <ChevronDown size={16} color={theme.text} />
        </TouchableOpacity>

        <Modal
          visible={visible}
          transparent={true}
          animationType="fade"
          onRequestClose={() => setVisible(false)}
        >
          <TouchableOpacity
            style={styles.dropdownOverlay}
            activeOpacity={1}
            onPress={() => setVisible(false)}
          >
            <Animated.View 
              entering={FadeIn.duration(150)}
              exiting={FadeOut.duration(100)}
              style={styles.dropdownAnimationContainer}
            >
              <BlurView 
                intensity={isDarkMode ? 15 : 40}
                tint={isDarkMode ? "dark" : "light"}
                style={styles.dropdownMenuBlur}
              >
                <View style={styles.dropdownHeader}>
                  <Text style={[styles.dropdownTitle, { color: theme.text }]}>
                    Select Image Mode
                  </Text>
                </View>
                
                <View style={styles.dropdownDivider} />
                
                <FlatList
                  data={options}
                  keyExtractor={(item) => item.id}
                  bounces={true}
                  showsVerticalScrollIndicator={false}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      style={[
                        styles.dropdownItem,
                        selectedOption.id === item.id ? 
                          { backgroundColor: 'rgba(0, 122, 255, 0.08)' } : {}
                      ]}
                      onPress={() => onSelect(item)}
                    >
                      <View style={styles.dropdownItemContent}>
                        <View style={[
                          styles.dropdownItemIconContainer,
                          { backgroundColor: selectedOption.id === item.id ? 'rgba(0, 122, 255, 0.8)' : 'rgba(140, 140, 140, 0.08)' }
                        ]}>
                          {item.icon && (
                            <item.icon 
                              size={16} 
                              color={selectedOption.id === item.id ? '#fff' : theme.text} 
                              strokeWidth={2}
                            />
                          )}
                        </View>
                        <Text style={[
                          styles.dropdownItemText,
                          { 
                            color: theme.text,
                            fontWeight: selectedOption.id === item.id ? '500' : '400'
                          }
                        ]}>
                          {item.label}
                        </Text>
                      </View>
                      {selectedOption.id === item.id && (
                        <CopyCheck size={16} color="rgba(0, 122, 255, 0.8)" strokeWidth={2} />
                      )}
                    </TouchableOpacity>
                  )}
                  style={styles.dropdownList}
                />
                
                <TouchableOpacity 
                  style={styles.dropdownCancelButton}
                  onPress={() => setVisible(false)}
                >
                  <Text style={styles.dropdownCancelText}>Cancel</Text>
                </TouchableOpacity>
              </BlurView>
            </Animated.View>
          </TouchableOpacity>
        </Modal>
      </>
    );
  };

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
        
        {/* Dropdowns for Model and Image Mode selection */}
        <View style={styles.dropdownsContainer}>
          <View style={styles.dropdownSection}>
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Select Model:</Text>
            {renderModelDropdown(
              modelDropdownVisible,
              setModelDropdownVisible,
              modelOptions,
              selectedModelOption,
              handleModelOptionSelect
            )}
          </View>
          
          <View style={styles.dropdownSection}>
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Image Mode:</Text>
            {renderImageModeDropdown(
              imageModeDropdownVisible,
              setImageModeDropdownVisible,
              imageModeOptions,
              selectedImageMode,
              handleImageModeSelect
            )}
          </View>
        </View>
        
        {/* Image preview - show current image or first from stack */}
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
          
          {/* Show count indicator if in stack mode and has multiple images */}
          {selectedImageMode.value === 'stack' && selectedImages.length > 0 && (
            <View style={styles.imageCountBadge}>
              <Text style={styles.imageCountText}>
                {selectedImages.length} {selectedImages.length === 1 ? 'image' : 'images'}
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
            <Text style={[styles.actionButtonText, { color: theme.text }]}>
              {selectedImageMode.value === 'stack' ? 'Select Images' : 'Gallery'}
            </Text>
          </TouchableOpacity>
        </View>
        
        {/* Process button */}
        <TouchableOpacity
          style={getStandardCardStyle({
            backgroundColor: theme.primary,
            marginBottom: 24,
            opacity: ((!selectedImage && selectedImages.length === 0) || processing) ? 0.7 : 1
          })}
          onPress={() => {
            console.log('Process button clicked!');
            handleProcessImage();
          }}
          disabled={(!selectedImage && selectedImages.length === 0) || processing}
        >
          {processing ? (
            <View style={styles.processingContainer}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.timerText}>{processingTimer}s</Text>
            </View>
          ) : (
            <Text style={styles.processButtonText}>
              Process {selectedImageMode.value === 'stack' ? 'Images' : 'Image'}
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
                  {result && result.usedNSGS && ' using NSGS'}
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
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
  },
  dropdownsContainer: {
    marginBottom: 24,
  },
  dropdownSection: {
    marginBottom: 16,
  },
  dropdownButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: BORDER_RADIUS,
    ...CARD_SHADOW,
  },
  dropdownSelectedItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dropdownSelectedText: {
    fontSize: 16,
    fontWeight: '500',
  },
  dropdownIcon: {
    marginRight: 10,
  },
  dropdownOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  dropdownAnimationContainer: {
    width: '90%',
    maxHeight: '70%',
    backgroundColor: 'transparent',
    borderRadius: 20,
    overflow: 'hidden',
  },
  dropdownMenuBlur: {
    width: '100%',
    height: '100%',
    borderRadius: 20,
    overflow: 'hidden',
    paddingBottom: 8,
  },
  dropdownHeader: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
    alignItems: 'center',
  },
  dropdownTitle: {
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
  },
  dropdownDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(140, 140, 140, 0.2)',
    marginHorizontal: 20,
  },
  dropdownList: {
    width: '100%',
    padding: 8,
    maxHeight: 300,
  },
  dropdownItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    marginVertical: 4,
    borderRadius: 12,
  },
  dropdownItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dropdownItemIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  dropdownItemText: {
    fontSize: 16,
  },
  dropdownCancelButton: {
    marginTop: 8,
    paddingVertical: 16,
    marginHorizontal: 20,
    borderRadius: 12,
    backgroundColor: 'rgba(60, 60, 67, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dropdownCancelText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
  },
  imageCountBadge: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  imageCountText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  apiConnectionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
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
});