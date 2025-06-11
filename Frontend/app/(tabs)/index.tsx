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
  Animated as RNAnimated,
  Dimensions
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, Upload, Image as ImageIcon, Wifi, Bug, BrainCircuit, Scan, Layers, Zap, Network, ChevronDown, LayoutGrid, CopyCheck } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
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
import * as FileSystem from 'expo-file-system';

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
  
  // New state for storing all processed images and slideshow controls
  const [processedResults, setProcessedResults] = useState<ProcessedImageResult[]>([]); // Store all processed results
  const [currentImageIndex, setCurrentImageIndex] = useState(0); // Current image index for slideshow
  
  const { theme, isDarkMode } = useTheme();
  
  // New state for tracking image processing progress
  const [processingProgress, setProcessingProgress] = useState({ current: 0, total: 0 });
  
  // Dropdown states
  const [modelDropdownVisible, setModelDropdownVisible] = useState(false);
  const [imageModeDropdownVisible, setImageModeDropdownVisible] = useState(false);
  const [selectedImageMode, setSelectedImageMode] = useState(imageModeOptions[0]);
  const [selectedModelOption, setSelectedModelOption] = useState(modelOptions[1]); // Default to segmentation without NSGS
  
  // Track if user has made explicit selections
  const [hasSelectedModel, setHasSelectedModel] = useState(false);
  const [hasSelectedImageMode, setHasSelectedImageMode] = useState(false);

  // New state for absolute position of dropdowns
  const [modelDropdownPosition, setModelDropdownPosition] = useState({ top: 0, left: 0, width: 0 });
  const [imageModeDropdownPosition, setImageModeDropdownPosition] = useState({ top: 0, left: 0, width: 0 });
  
  // Refs for measuring button positions
  const modelButtonRef = useRef<View>(null);
  const imageModeButtonRef = useRef<View>(null);

  // New state variables for NSGS visualization
  const [nsgsProcessingStats, setNsgsProcessingStats] = useState({
    graphNodes: 0,
    processedSpikes: 0,
    queueSize: 0,
    adaptationMultiplier: 1,
    processingTime: 0,
    status: '',
    logsOutput: '', // Add this field to store the raw logs
  });
  
  // Animation values for the progress indicators
  const graphNodesProgress = useRef(new RNAnimated.Value(0)).current;
  const spikesProgress = useRef(new RNAnimated.Value(0)).current;
  const queueProgress = useRef(new RNAnimated.Value(0)).current;
  const adaptationProgress = useRef(new RNAnimated.Value(0)).current;

  // Add state for triangle animation
  const [loadingDots, setLoadingDots] = useState<{x: number, y: number, opacity: number, fillPercent: number}[]>([]);
  const triangleAnimationRef = useRef<any>(null);

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
    let interval: any = null;
    
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
      const isConnected = await testApiConnection();
      setApiConnected(isConnected);
      if (!isConnected) {
        throw new Error("API connection failed. Please check your network and server settings.");
      }
    } catch (e: any) {
      setError(e.message || 'An unknown error occurred while connecting to the API');
      setApiConnected(false);
    } finally {
      setIsTestingConnection(false);
    }
  }, []);

  // Function to handle model option selection from dropdown
  const handleModelOptionSelect = (option: ModelOption) => {
    setSelectedModelOption(option);
    handleModelSelect(option.value, option.useNSGS);
    setModelDropdownVisible(false);
    setHasSelectedModel(true); // Mark that user has made a selection
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
  };

  // Function to handle model selection
  const handleModelSelect = useCallback((model: string, useNSGSFlag?: boolean) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedModel(model);
    // Only set NSGS if explicitly provided
    if (typeof useNSGSFlag === 'boolean') {
      setUseNSGS(useNSGSFlag);
      
      // Save NSGS preference to AsyncStorage
      try {
        AsyncStorage.setItem('useNSGS', JSON.stringify(useNSGSFlag));
        console.log(`Saved NSGS preference to AsyncStorage: ${useNSGSFlag}`);
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

  // Function to handle image mode selection from dropdown
  const handleImageModeSelect = (option: ImageModeOption) => {
    setSelectedImageMode(option);
    setImageModeDropdownVisible(false);
    setHasSelectedImageMode(true); // Mark that user has made a selection
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

  // Function to reset and start triangle animation
  const startTriangleAnimation = useCallback(() => {
    // Clear any existing animation
    if (triangleAnimationRef.current) {
      clearInterval(triangleAnimationRef.current);
    }
    
    // Initial empty state
    setLoadingDots([]);
    
    // Create D-shaped triangle dots array - tall left side, single dot at right
    const rows = 16; // More rows for taller left side
    const dots: {x: number, y: number, opacity: number, fillPercent: number}[] = [];
    
    // Function to generate D-shaped triangle pattern
    const generateDShapedTriangle = () => {
      const triangleDots: {x: number, y: number, opacity: number, fillPercent: number}[] = [];
      
      for (let row = 0; row < rows; row++) {
        // Calculate dots for this row to form a right triangle (D shape)
        // Top row has the most dots, bottom row has just one dot
        // This forms a diagonal line from top-right to bottom-left
        const dotsInThisRow = Math.ceil((rows - row) / 2);
        
        for (let col = 0; col < dotsInThisRow; col++) {
          triangleDots.push({
            x: col,
            y: row,
            opacity: 1, // Dots are always visible
            fillPercent: 0 // Start with 0% fill
          });
        }
      }
      return triangleDots;
    };
    
    const allDots = generateDShapedTriangle();
    setLoadingDots(allDots);
    
    // Animation timing variables
    let currentDot = 0;
    const totalDots = allDots.length;
    
    // Start animation interval - even slower (200ms)
    triangleAnimationRef.current = setInterval(() => {
      if (currentDot < totalDots) {
        setLoadingDots(prev => {
          const newDots = [...prev];
          // Increment the fill percentage of the current dot
          newDots[currentDot] = {...newDots[currentDot], fillPercent: 1};
          return newDots;
        });
        currentDot++;
      } else {
        // Reset the animation to create a continuous effect
        currentDot = 0;
        
        // Start filling again from the beginning with a longer delay
        setTimeout(() => {
          setLoadingDots(prev => prev.map(dot => ({...dot, fillPercent: 0})));
        }, 1200);
      }
    }, 200); // Slower animation
    
    return () => {
      if (triangleAnimationRef.current) {
        clearInterval(triangleAnimationRef.current);
      }
    };
  }, []);

  // Update the simulation function to use triangle animation instead
  useEffect(() => {
    if (processing && useNSGS) {
      // Reset progress values
      graphNodesProgress.setValue(0);
      spikesProgress.setValue(0);
      queueProgress.setValue(0);
      adaptationProgress.setValue(0);
      
      // Initial state
      setNsgsProcessingStats({
        graphNodes: 0,
        processedSpikes: 0,
        queueSize: 0,
        adaptationMultiplier: 1,
        processingTime: 0,
        status: 'Initializing NSGS...',
        logsOutput: '', // Initialize empty logs
      });
      
      // Start triangle animation
      startTriangleAnimation();
    } else {
      // Clear animation when not processing
      if (triangleAnimationRef.current) {
        clearInterval(triangleAnimationRef.current);
      }
    }
  }, [processing, useNSGS, startTriangleAnimation]);

  // Move this before renderNsgsMetrics
  // Add function to fetch NSGS logs directly from debug endpoint
  const fetchNsgsDebugLogs = async () => {
    try {
      const response = await fetch(`${getApiUrl()}/debug/nsgs-logs`);
      const data = await response.json();
      
      if (data && data.logs) {
        // Update the stats directly with the fetched logs
        setNsgsProcessingStats(prevStats => ({
          ...prevStats,
          logsOutput: data.logs,
          status: data.stats?.status || prevStats.status
        }));
        console.log("Fetched NSGS logs directly from debug endpoint");
      }
    } catch (error) {
      console.error("Error fetching NSGS debug logs:", error);
    }
  };

  // Add rendered metrics to show below the result
  const renderNsgsMetrics = useCallback(() => {
    if (!processedImage || !useNSGS) return null;
    
    return (
      <View style={[styles.nsgsMetricsContainer, {
        backgroundColor: theme.card,
        ...(isDarkMode ? {} : CARD_SHADOW),
        marginBottom: 20, // Reduced margin since we'll have a separate logs container
      }]}>
        <Text style={[styles.nsgsMetricsTitle, { color: theme.text }]}>NSGS Processing Details</Text>
        
        {/* Add debug refresh button */}
        <TouchableOpacity 
          style={[styles.debugRefreshButton, {
            backgroundColor: isDarkMode ? '#333' : '#e0e0e0',
            marginVertical: 10,
          }]}
          onPress={fetchNsgsDebugLogs}
        >
          <Text style={[styles.debugRefreshText, { color: isDarkMode ? '#fff' : '#333' }]}>
            Refresh Logs
          </Text>
        </TouchableOpacity>
        
        {/* NSGS Log Status Indicator */}
        <View style={styles.logStatusContainer}>
          <Text style={[styles.logStatusText, { color: theme.text }]}>
            Log Status: {nsgsProcessingStats.logsOutput && nsgsProcessingStats.logsOutput.length > 0 
              ? <Text style={{color: 'green'}}>Available</Text> 
              : <Text style={{color: 'red'}}>Not Available</Text>}
          </Text>
          <Text style={[styles.logStatusText, { color: theme.text }]}>
            Log Length: {nsgsProcessingStats.logsOutput ? nsgsProcessingStats.logsOutput.length : 0} characters
          </Text>
        </View>
        
        {/* Remove the logs ScrollView from here - it will be moved to a separate container */}
      </View>
    );
  }, [processedImage, useNSGS, nsgsProcessingStats, theme, isDarkMode, fetchNsgsDebugLogs]);

  // Now create a new separate function to render just the logs
  const renderNsgsLogs = useCallback(() => {
    if (!processedImage || !useNSGS) return null;
    
    return (
      <View style={[styles.standaloneLogsContainer, {
        backgroundColor: theme.card,
        ...(isDarkMode ? {} : CARD_SHADOW),
        borderWidth: 2,
        borderColor: theme.primary,
        marginTop: 30,
      }]}>
        <Text style={[styles.nsgsLogsTitle, { 
          color: theme.text,
          fontSize: 20,
          marginBottom: 15
        }]}>
          NSGS Full Logs
        </Text>
        
        <TouchableOpacity 
          style={[styles.debugRefreshButton, {
            backgroundColor: theme.primary,
            marginVertical: 10,
            width: '50%',
            alignSelf: 'center',
          }]}
          onPress={fetchNsgsDebugLogs}
        >
          <Text style={[styles.debugRefreshText, { color: '#fff' }]}>
            Refresh Logs
          </Text>
        </TouchableOpacity>
        
        {/* NSGS Raw Log Output in a standalone container */}
        <ScrollView 
          style={[styles.nsgsLogsScrollView, { 
            backgroundColor: isDarkMode ? '#1a1a1a' : '#f0f0f0',
            height: 1000, // Large fixed height
            minHeight: 800, 
            borderWidth: 2,
            borderColor: isDarkMode ? theme.primary + '80' : theme.primary + '40',
            padding: 15,
            marginBottom: 30,
            marginTop: 10,
            borderRadius: 8,
          }]}
          showsVerticalScrollIndicator={true}
          nestedScrollEnabled={true}
        >
          <Text style={[styles.nsgsLogsText, { 
            color: isDarkMode ? '#e0e0e0' : '#333333',
            fontSize: 13,
            lineHeight: 18,
            fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
            paddingBottom: 30,
          }]}>
            {nsgsProcessingStats.logsOutput && nsgsProcessingStats.logsOutput.length > 0 
              ? nsgsProcessingStats.logsOutput 
              : "No logs available - Press 'Refresh Logs' button to fetch the latest logs"}
          </Text>
        </ScrollView>
      </View>
    );
  }, [processedImage, useNSGS, nsgsProcessingStats, theme, isDarkMode]);

  // Render the triangle loading animation
  const renderTriangleLoading = () => {
    if (!processing || !useNSGS) return null;
    
    const dotSize = 12;  // Even larger dots
    const spacing = 20;  // More spacing
    
    return (
      <View style={styles.triangleLoadingContainer}>
        <Text style={styles.triangleLoadingTitle}>NSGS Neural Processing</Text>
        
        <Text style={styles.triangleLoadingStatus}>{nsgsProcessingStats.status || "Processing..."}</Text>
        
        <View style={styles.triangleWrapper}>
          {loadingDots.map((dot, index) => (
            <View
              key={index}
              style={{
                position: 'absolute',
                left: dot.x * spacing,
                top: dot.y * spacing,
                width: dotSize,
                height: dotSize,
                borderRadius: dotSize / 2,
                borderWidth: 2,
                borderColor: '#9A6AFF',
                justifyContent: 'center',
                alignItems: 'center',
                overflow: 'hidden',
              }}
            >
              {/* Inner filled circle that grows based on fillPercent */}
              <RNAnimated.View
                style={{
                  width: dot.fillPercent * (dotSize - 5),
                  height: dot.fillPercent * (dotSize - 5),
                  borderRadius: (dotSize - 5) / 2,
                  backgroundColor: '#6A35D9',
                  opacity: 0.9,
                }}
              />
            </View>
          ))}
        </View>
        
        {processingTimer > 0 && (
          <Text style={styles.triangleLoadingTimer}>{processingTimer}s</Text>
        )}
      </View>
    );
  };

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
    
    // Reset NSGS stats
    setNsgsProcessingStats({
      graphNodes: 0,
      processedSpikes: 0,
      queueSize: 0,
      adaptationMultiplier: 1,
      processingTime: 0,
      status: 'Initializing NSGS...',
      logsOutput: '', // Initialize empty logs
    });
    
    const startTime = Date.now();
    
    try {
      // Log the image info before processing
      console.log(`Processing ${selectedImages.length} images with model: ${selectedModel}`);
      
      // Log the API URL being used
      const apiUrl = getApiUrl();
      console.log(`API URL: ${apiUrl}`);
      
      // Trigger success haptic when processing starts
      triggerHapticNotification(Haptics.NotificationFeedbackType.Success);
      
      let finalResult: ProcessedImageResult | null = null;
      const processedResults: ProcessedImageResult[] = [];
      
      // Create variables to accumulate NSGS stats across all processed images
      let totalGraphNodes = 0;
      let totalProcessedSpikes = 0;
      let maxQueueSize = 0;
      let finalAdaptationMultiplier = 1;
      let nsgsProcessingTime = 0; // Renamed from totalProcessingTime to avoid duplicate
      
      // Generate a unique stackId for this batch of images if using stack mode
      const isStackMode = selectedImageMode.value === 'stack' && selectedImages.length > 1;
      const stackId = isStackMode ? `stack_${Date.now()}` : undefined;
      
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
          
          const result = await debugUploadImage(selectedImage || selectedImages[0]);
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
          
          // Process all images in the stack or just the single selected image
          const imagesToProcess = selectedImageMode.value === 'stack' ? selectedImages : [selectedImage];
          
          // Set total images to process for progress tracking
          setProcessingProgress({ current: 0, total: imagesToProcess.length });
          
          for (let i = 0; i < imagesToProcess.length; i++) {
            const currentImage = imagesToProcess[i];
            if (!currentImage) continue;
            
            // Update processing progress
            setProcessingProgress({ current: i + 1, total: imagesToProcess.length });
            
            try {
              const result = await processImage(currentImage, selectedModel, useNSGS);
              
              // Update with real NSGS stats if available
              if (useNSGS && result && (result as any).nsgsStats) {
                const nsgsStats = (result as any).nsgsStats;
                console.log('Received NSGS stats from backend:', nsgsStats);
                
                // Accumulate stats across all processed images
                totalGraphNodes = Math.max(totalGraphNodes, nsgsStats.graphNodes || 6400);
                totalProcessedSpikes += (nsgsStats.processedSpikes || 0);
                maxQueueSize = Math.max(maxQueueSize, nsgsStats.queueSize || 0);
                finalAdaptationMultiplier = nsgsStats.adaptationMultiplier || finalAdaptationMultiplier;
                nsgsProcessingTime += (nsgsStats.processingTime || 0); // Use renamed variable
                
                // Store the raw logs if available
                const nsgsLogs = nsgsStats.logsOutput || '';
                console.log(`NSGS logs in response: ${nsgsLogs.substring(0, 100)}${nsgsLogs.length > 100 ? '...' : ''}`);
                console.log(`NSGS logs length: ${nsgsLogs.length}`);
                
                // Update current stats for the progress display
                setNsgsProcessingStats({
                  graphNodes: totalGraphNodes,
                  processedSpikes: totalProcessedSpikes,
                  queueSize: maxQueueSize,
                  adaptationMultiplier: finalAdaptationMultiplier,
                  processingTime: nsgsProcessingTime, // Use renamed variable
                  status: nsgsStats.status || 'Processing complete',
                  logsOutput: nsgsLogs, // Store the logs
                });
              }
              
              // Add stack information if this is a stack of images
              if (isStackMode) {
                (result as ProcessedImageResult).stackId = stackId;
                (result as ProcessedImageResult).isStackImage = true;
              }
              
              processedResults.push(result);
              console.log(`Processed image ${i + 1}/${imagesToProcess.length}:`, result);
            } catch (imageError) {
              console.error(`Error processing image ${i + 1}:`, imageError);
              // Continue with next image
            }
          }
          
          // After all images are processed, set the final accumulated NSGS stats
          if (useNSGS) {
            setNsgsProcessingStats({
              graphNodes: totalGraphNodes,
              processedSpikes: totalProcessedSpikes,
              queueSize: maxQueueSize,
              adaptationMultiplier: finalAdaptationMultiplier,
              processingTime: nsgsProcessingTime, // Use renamed variable
              status: 'All processing complete',
              logsOutput: '', // Clear logs for final result
            });
          }
          
          // Store all processed results for slideshow
          setProcessedResults(processedResults);
          setCurrentImageIndex(0);
          
          // Use the last result as the final one to display initially
          finalResult = processedResults[processedResults.length - 1];
          
          // Calculate actual UI processing time
          const endTime = Date.now();
          const uiProcessingTime = Math.round((endTime - startTime) / 1000);
          setProcessingTime(uiProcessingTime);
          
          // Store all results if needed
          setResult(finalResult);
          
          // Show result container
          setIsResultVisible(true);
          
          // If processing multiple images, show a summary
          if (processedResults.length > 1) {
            Alert.alert('Processing Complete', `Successfully processed ${processedResults.length} images!`);
          }
        } catch (processingError: any) {
          console.error('Image processing error:', processingError);
          throw new Error(`Processing failed: ${processingError.message}`);
        }
        
        // Force refresh the image cache by adding a timestamp parameter
        const timestamp = new Date().getTime();
        
        // First, prefer base64 image data from the response if available
        if (finalResult && finalResult.imageBase64) {
          console.log('Using base64 image data directly from server response');
          setProcessedImage(finalResult.imageBase64);
        }
        // For mobile platforms, prefer local cached image if available
        else if (finalResult && (finalResult.cachedImageUrl || finalResult.localUri)) {
          const localUri = finalResult.cachedImageUrl || finalResult.localUri || '';
          console.log(`Using locally cached image: ${localUri}`);
          if (localUri) {
            setProcessedImage(localUri);
          } else {
            console.warn('Local URI is empty, falling back to remote URL');
          }
        }
        // Check for fullUrl in the response first
        else if (finalResult && finalResult.fullUrl) {
          console.log(`Using full URL from server: ${finalResult.fullUrl}`);
          // Always add a cache-busting parameter
          const cachebustedUrl = finalResult.fullUrl.includes('?') 
            ? `${finalResult.fullUrl}&t=${timestamp}` 
            : `${finalResult.fullUrl}?t=${timestamp}`;
          
          try {
            // On mobile, try to download and display as base64 for more reliability
            if (Platform.OS !== 'web') {
              console.log('Attempting to convert remote image to base64 for display');
              const tempDir = FileSystem.cacheDirectory;
              const tempFile = `${tempDir}temp_${timestamp}.jpg`;
              
              // Download to temporary file
              const downloadResult = await FileSystem.downloadAsync(cachebustedUrl, tempFile);
              if (downloadResult.status === 200) {
                // Read the file as base64
                const base64 = await FileSystem.readAsStringAsync(tempFile, {
                  encoding: FileSystem.EncodingType.Base64
                });
                
                if (base64) {
                  const dataUri = `data:image/jpeg;base64,${base64}`;
                  console.log('Successfully converted image to base64');
                  setProcessedImage(dataUri);
                  return; // Stop here since we've set the image
                }
              }
            }
          } catch (e) {
            console.error('Error converting image to base64:', e);
          }
          
          // Fall back to the remote URL if base64 conversion fails
          setProcessedImage(cachebustedUrl);
        }
        // If no fullUrl, use processedImageUrl with API URL
        else if (finalResult && finalResult.processedImageUrl) {
          const fullUrl = `${apiUrl}${finalResult.processedImageUrl}`;
          console.log(`Constructed URL from processedImageUrl: ${fullUrl}`);
          // Always add a cache-busting parameter
          const cachebustedUrl = fullUrl.includes('?') 
            ? `${fullUrl}&t=${timestamp}` 
            : `${fullUrl}?t=${timestamp}`;
          
          try {
            // On mobile, try to download and display as base64 for more reliability
            if (Platform.OS !== 'web') {
              console.log('Attempting to convert remote image to base64 for display');
              const tempDir = FileSystem.cacheDirectory;
              const tempFile = `${tempDir}temp_${timestamp}.jpg`;
              
              // Download to temporary file
              const downloadResult = await FileSystem.downloadAsync(cachebustedUrl, tempFile);
              if (downloadResult.status === 200) {
                // Read the file as base64
                const base64 = await FileSystem.readAsStringAsync(tempFile, {
                  encoding: FileSystem.EncodingType.Base64
                });
                
                if (base64) {
                  const dataUri = `data:image/jpeg;base64,${base64}`;
                  console.log('Successfully converted image to base64');
                  setProcessedImage(dataUri);
                  return; // Stop here since we've set the image
                }
              }
            }
          } catch (e) {
            console.error('Error converting image to base64:', e);
          }
          
          // Fall back to the remote URL if base64 conversion fails
          setProcessedImage(cachebustedUrl);
        } else {
          console.error('No image URL in response:', finalResult);
          throw new Error('No processed image URL received from server');
        }
        
        // Show success message (or warning for fallback)
        if (finalResult && finalResult.isFallback) {
          Alert.alert(
            'Processing Warning', 
            'The segmentation model failed to process this image, so we\'re showing the original image. Try the object detection model instead.'
          );
        } else if (finalResult && finalResult.warning && finalResult.warning.includes('fallback detection')) {
          Alert.alert(
            'Segmentation Notice', 
            'The segmentation model had issues, so we used object detection instead. The results are shown with bounding boxes rather than full segmentation masks.'
          );
        } else if (finalResult && finalResult.warning) {
          Alert.alert(
            'Processing Notice', 
            `The image was processed with a warning: ${finalResult.warning}`
          );
        } else if (finalResult && finalResult.message && finalResult.message.includes('best match')) {
          Alert.alert(
            'Processing Success',
            'The image was processed successfully, but we had to use the best matching output file.'
          );
        }
      }
      
      // Trigger success haptic when processing is complete
      triggerHapticNotification(Haptics.NotificationFeedbackType.Success);
    } catch (err: any) {
      console.error('Error processing image:', err);
      const errorMessage = err?.message || 'Failed to process image. Please try debug mode to troubleshoot.';
      setError(errorMessage);
      
      // Make sure the result view is shown
      setIsResultVisible(true);
      
      // Show error dialog with more details
      Alert.alert(
        'Processing Error', 
        `Error: ${errorMessage}\n\nTry enabling Debug mode to test the connection.`
      );
      
      // Trigger error haptic
      triggerHapticNotification(Haptics.NotificationFeedbackType.Error);
    } finally {
      // Stop triangle animation
      if (triangleAnimationRef.current) {
        clearInterval(triangleAnimationRef.current);
      }
      
      setProcessing(false);
      // Reset progress
      setProcessingProgress({ current: 0, total: 0 });
      
      // Add automatic log fetching when NSGS is used
      if (useNSGS) {
        // Wait a brief moment to ensure logs are saved on the backend
        setTimeout(() => {
          console.log("Auto-fetching NSGS logs after processing");
          fetchNsgsDebugLogs();
        }, 500);
      }
    }
  }, [selectedImage, selectedImages, selectedModel, debugMode, useNSGS, selectedImageMode, startTriangleAnimation]);

  // Toggle debug mode
  const toggleDebugMode = useCallback(() => {
    setDebugMode(!debugMode);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
  }, [debugMode]);

  // Function to show model dropdown with correct position
  const showModelDropdown = () => {
    if (modelButtonRef.current) {
      modelButtonRef.current.measure((x: number, y: number, width: number, height: number, pageX: number, pageY: number) => {
        setModelDropdownPosition({
          top: pageY + height + 5,
          left: pageX,
          width: width
        });
        setModelDropdownVisible(true);
      });
    }
  };
  
  // Function to show image mode dropdown with correct position
  const showImageModeDropdown = () => {
    if (imageModeButtonRef.current) {
      imageModeButtonRef.current.measure((x: number, y: number, width: number, height: number, pageX: number, pageY: number) => {
        setImageModeDropdownPosition({
          top: pageY + height + 5,
          left: pageX,
          width: width
        });
        setImageModeDropdownVisible(true);
      });
    }
  };

  // Render dropdown for model selection
  const renderModelDropdown = () => {
    const backgroundColor = hasSelectedModel && selectedModelOption.color ? selectedModelOption.color : theme.card;
    
    return (
      <>
        <TouchableOpacity
          ref={modelButtonRef}
          style={[
            styles.dropdownButton,
            { backgroundColor: backgroundColor }
          ]}
          onPress={showModelDropdown}
        >
          <View style={styles.dropdownSelectedItem}>
            {selectedModelOption.icon && hasSelectedModel ? (
              <selectedModelOption.icon 
                size={20} 
                color={hasSelectedModel && selectedModelOption.color ? '#fff' : theme.text} 
                style={styles.dropdownIcon} 
              />
            ) : null}
            <Text style={[
              styles.dropdownSelectedText, 
              { color: hasSelectedModel && selectedModelOption.color ? '#fff' : theme.text }
            ]}>
              {hasSelectedModel ? selectedModelOption.label : "Select Model"}
            </Text>
          </View>
          <ChevronDown size={16} color={hasSelectedModel && selectedModelOption.color ? '#fff' : theme.text} />
        </TouchableOpacity>
      </>
    );
  };

  // Render dropdown for image mode selection
  const renderImageModeDropdown = () => {
    return (
      <>
        <TouchableOpacity
          ref={imageModeButtonRef}
          style={[
            styles.dropdownButton,
            { backgroundColor: theme.card }
          ]}
          onPress={showImageModeDropdown}
        >
          <View style={styles.dropdownSelectedItem}>
            {selectedImageMode.icon && hasSelectedImageMode ? (
              <selectedImageMode.icon 
                size={20} 
                color={theme.text} 
                style={styles.dropdownIcon} 
              />
            ) : null}
            <Text style={[
              styles.dropdownSelectedText, 
              { color: theme.text }
            ]}>
              {hasSelectedImageMode ? selectedImageMode.label : "Select Image Mode"}
            </Text>
          </View>
          <ChevronDown size={16} color={theme.text} />
        </TouchableOpacity>
      </>
    );
  };

  // Render model dropdown list
  const renderModelDropdownList = () => {
    if (!modelDropdownVisible) return null;
    
    return (
      <View
        style={[
          StyleSheet.absoluteFill,
          styles.dropdownBackdrop
        ]}
      >
        <TouchableOpacity
          style={[
            StyleSheet.absoluteFill,
          ]}
          activeOpacity={1}
          onPress={() => setModelDropdownVisible(false)}
        />
        
        <View 
          style={[
            styles.dropdownPopup,
            {
              position: 'absolute',
              top: modelDropdownPosition.top,
              left: modelDropdownPosition.left,
              width: modelDropdownPosition.width,
              backgroundColor: isDarkMode ? 'rgba(40, 40, 40, 0.98)' : 'rgba(255, 255, 255, 0.98)',
              zIndex: 1001,
            }
          ]}
        >
          <View style={styles.dropdownHeader}>
            <Text style={[styles.dropdownTitle, { color: theme.text }]}>
              Select Model
            </Text>
          </View>
          
          <View style={styles.dropdownDivider} />
          
          <ScrollView style={styles.dropdownScrollView}>
            {modelOptions.map((item) => (
              <TouchableOpacity
                key={item.id}
                style={[
                  styles.dropdownItem,
                  selectedModelOption.id === item.id ? 
                    (item.color ? { backgroundColor: `${item.color}20` } : { backgroundColor: 'rgba(0, 122, 255, 0.08)' }) : {}
                ]}
                onPress={() => {
                  handleModelOptionSelect(item);
                  setModelDropdownVisible(false);
                }}
              >
                <View style={styles.dropdownItemContent}>
                  <View style={[
                    styles.dropdownItemIconContainer,
                    { backgroundColor: selectedModelOption.id === item.id ? (item.color || 'rgba(0, 122, 255, 0.8)') : 'rgba(140, 140, 140, 0.08)' }
                  ]}>
                    {item.icon && (
                      <item.icon 
                        size={16} 
                        color={selectedModelOption.id === item.id ? '#fff' : theme.text} 
                        strokeWidth={2}
                      />
                    )}
                  </View>
                  <Text style={[
                    styles.dropdownItemText,
                    { 
                      color: theme.text,
                      fontWeight: selectedModelOption.id === item.id ? '600' : '400'
                    }
                  ]}>
                    {item.label}
                  </Text>
                </View>
                {selectedModelOption.id === item.id && (
                  <CopyCheck size={16} color={item.color || 'rgba(0, 122, 255, 0.8)'} strokeWidth={2} />
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>
          
          <TouchableOpacity 
            style={styles.dropdownCancelButton}
            onPress={() => setModelDropdownVisible(false)}
          >
            <Text style={styles.dropdownCancelText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Render image mode dropdown list
  const renderImageModeDropdownList = () => {
    if (!imageModeDropdownVisible) return null;
    
    return (
      <View
        style={[
          StyleSheet.absoluteFill,
          styles.dropdownBackdrop
        ]}
      >
        <TouchableOpacity
          style={[
            StyleSheet.absoluteFill,
          ]}
          activeOpacity={1}
          onPress={() => setImageModeDropdownVisible(false)}
        />
        
        <View 
          style={[
            styles.dropdownPopup,
            {
              position: 'absolute',
              top: imageModeDropdownPosition.top,
              left: imageModeDropdownPosition.left,
              width: imageModeDropdownPosition.width,
              backgroundColor: isDarkMode ? 'rgba(40, 40, 40, 0.98)' : 'rgba(255, 255, 255, 0.98)',
              zIndex: 1001,
            }
          ]}
        >
          <View style={styles.dropdownHeader}>
            <Text style={[styles.dropdownTitle, { color: theme.text }]}>
              Select Image Mode
            </Text>
          </View>
          
          <View style={styles.dropdownDivider} />
          
          <ScrollView style={styles.dropdownScrollView}>
            {imageModeOptions.map((item) => (
              <TouchableOpacity
                key={item.id}
                style={[
                  styles.dropdownItem,
                  selectedImageMode.id === item.id ? 
                    { backgroundColor: 'rgba(0, 122, 255, 0.08)' } : {}
                ]}
                onPress={() => {
                  handleImageModeSelect(item);
                  setImageModeDropdownVisible(false);
                }}
              >
                <View style={styles.dropdownItemContent}>
                  <View style={[
                    styles.dropdownItemIconContainer,
                    { backgroundColor: selectedImageMode.id === item.id ? 'rgba(0, 122, 255, 0.8)' : 'rgba(140, 140, 140, 0.08)' }
                  ]}>
                    {item.icon && (
                      <item.icon 
                        size={16} 
                        color={selectedImageMode.id === item.id ? '#fff' : theme.text} 
                        strokeWidth={2}
                      />
                    )}
                  </View>
                  <Text style={[
                    styles.dropdownItemText,
                    { 
                      color: theme.text,
                      fontWeight: selectedImageMode.id === item.id ? '600' : '400'
                    }
                  ]}>
                    {item.label}
                  </Text>
                </View>
                {selectedImageMode.id === item.id && (
                  <CopyCheck size={16} color="rgba(0, 122, 255, 0.8)" strokeWidth={2} />
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>
          
          <TouchableOpacity 
            style={styles.dropdownCancelButton}
            onPress={() => setImageModeDropdownVisible(false)}
          >
            <Text style={styles.dropdownCancelText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Custom NSGS processing visualization
  const renderNsgsProcessingUI = () => {
    if (!processing || !useNSGS) return null;
    
    return (
      <View style={styles.nsgsProcessingContainer}>
        <Text style={styles.nsgsHeader}>NSGS Neural Processing</Text>
        
        <Text style={styles.nsgsStatusText}>{nsgsProcessingStats.status || "Processing..."}</Text>
        
        {processingTimer > 0 && (
          <Text style={styles.nsgsProcessingTime}>
            Time Elapsed: {processingTimer}s
          </Text>
        )}
      </View>
    );
  };

  // Update the Process button to use triangle loading animation
  const renderProcessButton = () => {
    return (
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
          useNSGS ? (
            renderTriangleLoading()
          ) : (
            <View style={styles.processingContainer}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.timerText}>{processingTimer}s</Text>
              {processingProgress.total > 1 && (
                <Text style={styles.imageProgressText}>
                  {processingProgress.current}/{processingProgress.total} images
                </Text>
              )}
            </View>
          )
        ) : (
          <Text style={styles.processButtonText}>
            Process {selectedImageMode.value === 'stack' ? 'Images' : 'Image'}
          </Text>
        )}
      </TouchableOpacity>
    );
  };

  // Function to navigate to the next image in the slideshow
  const goToNextImage = useCallback(() => {
    if (processedResults.length > 1 && currentImageIndex < processedResults.length - 1) {
      const nextIndex = currentImageIndex + 1;
      setCurrentImageIndex(nextIndex);
      
      // Update the displayed image
      const nextResult = processedResults[nextIndex];
      updateDisplayedImage(nextResult);
      
      // Provide haptic feedback
      triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    }
  }, [processedResults, currentImageIndex]);

  // Function to navigate to the previous image in the slideshow
  const goToPrevImage = useCallback(() => {
    if (processedResults.length > 1 && currentImageIndex > 0) {
      const prevIndex = currentImageIndex - 1;
      setCurrentImageIndex(prevIndex);
      
      // Update the displayed image
      const prevResult = processedResults[prevIndex];
      updateDisplayedImage(prevResult);
      
      // Provide haptic feedback
      triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    }
  }, [processedResults, currentImageIndex]);

  // Helper function to update the displayed image based on a result
  const updateDisplayedImage = useCallback((result: ProcessedImageResult) => {
    if (result) {
      // Set the image based on the same priority as in handleProcessImage
      if (result.imageBase64) {
        setProcessedImage(result.imageBase64);
      } else if (result.cachedImageUrl || result.localUri) {
        setProcessedImage(result.cachedImageUrl || result.localUri || '');
      } else if (result.fullUrl) {
        setProcessedImage(result.fullUrl);
      } else if (result.processedImageUrl) {
        setProcessedImage(`${getApiUrl()}${result.processedImageUrl}`);
      }
      
      // Update the current result for metadata display
      setResult(result);
    }
  }, []);

  // Add a useEffect to fetch logs on component mount
  useEffect(() => {
    // Fetch NSGS logs when the component mounts
    if (useNSGS) {
      console.log("Fetching NSGS logs on component mount");
      fetchNsgsDebugLogs();
    }
  }, [useNSGS]); // Only re-run if useNSGS changes

  return (
    <SafeAreaView style={{ 
      flex: 1, 
      backgroundColor: theme.background,
      minHeight: Dimensions.get('window').height, // Ensure minimum height based on window
    }}>
      <ScrollView 
        contentContainerStyle={[
          styles.container,
          { 
            paddingBottom: Platform.OS === 'ios' ? 600 : 580, // Even more padding at the bottom
          }
        ]}
        showsVerticalScrollIndicator={true} // Ensure scroll indicator is visible
      >
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
            {renderModelDropdown()}
          </View>
          
          <View style={styles.dropdownSection}>
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Image Mode:</Text>
            {renderImageModeDropdown()}
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
        {renderProcessButton()}
        
        {/* Results area (show processed image or error) */}
        {isResultVisible && (
          <Animated.View 
            style={[
              styles.resultsContainer,
              { backgroundColor: isDarkMode ? theme.card : '#fff' }
            ]}
            entering={FadeIn.duration(500)}
          >
            {processing ? (
              // Show processing UI based on model type
              useNSGS ? renderTriangleLoading() : (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator size="large" color={theme.primary} />
                  <Text style={[styles.loadingText, { color: theme.textSecondary }]}>
                    Processing image... {processingTimer > 0 ? `(${processingTimer}s)` : ''}
                  </Text>
                  
                  {processingProgress.total > 1 && (
                    <Text style={[styles.progressText, { color: theme.textSecondary }]}>
                      Image {processingProgress.current}/{processingProgress.total}
                    </Text>
                  )}
                </View>
              )
            ) : error ? (
              <View style={styles.errorContainer}>
                <Text style={[styles.errorText, { color: theme.error }]}>{error}</Text>
                
                <TouchableOpacity
                  style={[styles.reconnectButton, { backgroundColor: theme.error }]}
                  onPress={checkApiConnection}
                >
                  <Text style={styles.reconnectButtonText}>Test Connection</Text>
                </TouchableOpacity>
              </View>
            ) : processedImage ? (
              <View style={[styles.resultImageContainer, { backgroundColor: theme.card }]}>
                {/* Image title when showing a stack slideshow */}
                {processedResults.length > 1 && (
                  <View style={styles.imageTitleContainer}>
                    <Text style={styles.imageTitleText}>
                      {result?.originalImageName || `Image ${currentImageIndex + 1}`}
                    </Text>
                  </View>
                )}
                
                <Image
                  source={{ uri: processedImage }}
                  style={styles.resultImage}
                  resizeMode="contain"
                />

                {/* Processing Time Display */}
                {processingTime !== null && (
                  <View style={{
                    flexDirection: 'row',
                    alignItems: 'center',
                    marginTop: 12,
                    padding: 8,
                    borderRadius: 12,
                    backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                  }}>
                    <Zap size={16} color={theme.text} style={{ marginRight: 8 }} />
                    <Text style={{
                      fontSize: 14,
                      fontWeight: '500',
                      color: theme.text
                    }}>
                      Processing Time: {processingTime}s
                    </Text>
                  </View>
                )}
                
                {/* Slideshow Navigation - Show only when we have multiple images */}
                {processedResults.length > 1 && (
                  <View style={styles.slideshowControls}>
                    {/* Previous button */}
                    <TouchableOpacity 
                      style={[
                        styles.slideshowButton, 
                        currentImageIndex === 0 && styles.slideshowButtonDisabled
                      ]}
                      onPress={goToPrevImage}
                      disabled={currentImageIndex === 0}
                    >
                      <Text style={styles.slideshowButtonText}>◀</Text>
                    </TouchableOpacity>
                    
                    {/* Image counter */}
                    <View style={styles.slideshowCounter}>
                      <Text style={styles.slideshowCounterText}>
                        {currentImageIndex + 1} / {processedResults.length}
                      </Text>
                      
                      {/* Dot indicators for slideshow */}
                      <View style={styles.dotIndicatorContainer}>
                        {processedResults.map((_, index) => (
                          <View 
                            key={index} 
                            style={[
                              styles.dotIndicator,
                              index === currentImageIndex && styles.dotIndicatorActive
                            ]} 
                          />
                        ))}
                      </View>
                    </View>
                    
                    {/* Next button */}
                    <TouchableOpacity 
                      style={[
                        styles.slideshowButton,
                        currentImageIndex === processedResults.length - 1 && styles.slideshowButtonDisabled
                      ]}
                      onPress={goToNextImage}
                      disabled={currentImageIndex === processedResults.length - 1}
                    >
                      <Text style={styles.slideshowButtonText}>▶</Text>
                    </TouchableOpacity>
                  </View>
                )}
              </View>
            ) : null}
            
            {/* NSGS Metrics Container */}
            {useNSGS && processedImage && (
              <>
                <View style={[styles.standalonensgsMetricsContainer, { 
                  backgroundColor: theme.card,
                  paddingBottom: 20,
                  marginBottom: 20, // Reduced margin since we'll have a separate logs container
                }]}>
                  <Text style={[styles.nsgsMetricsHeaderText, { color: theme.text }]}>NSGS Processing Details</Text>
                  <ScrollView 
                    style={styles.nsgsMetricsScrollView}
                    contentContainerStyle={{ paddingBottom: 20 }}
                    showsVerticalScrollIndicator={true}
                    nestedScrollEnabled={true}
                  >
                    {renderNsgsMetrics()}
                  </ScrollView>
                </View>
                
                {/* Separate standalone container for logs */}
                {renderNsgsLogs()}
              </>
            )}
          </Animated.View>
        )}
      </ScrollView>

      {/* Render dropdown lists */}
      {renderModelDropdownList()}
      {renderImageModeDropdownList()}
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
  dropdownMenuContent: {
    width: '100%',
    height: '100%',
    borderRadius: 20,
    overflow: 'hidden',
    paddingBottom: 8,
  },
  dropdownBackdrop: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    zIndex: 1000,
  },
  dropdownPopup: {
    borderRadius: BORDER_RADIUS,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    maxHeight: 400,
  },
  dropdownHeader: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
    alignItems: 'center',
  },
  dropdownTitle: {
    fontSize: 17,
    fontWeight: '600',
    textAlign: 'center',
  },
  dropdownDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(140, 140, 140, 0.2)',
    marginHorizontal: 10,
  },
  dropdownScrollView: {
    maxHeight: 250,
    paddingHorizontal: 6,
    paddingTop: 6,
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
    paddingVertical: 12,
    marginVertical: 3,
    borderRadius: 10,
  },
  dropdownItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dropdownItemIconContainer: {
    width: 34,
    height: 34,
    borderRadius: 17,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  dropdownItemText: {
    fontSize: 16,
  },
  dropdownCancelButton: {
    marginVertical: 10,
    paddingVertical: 14,
    marginHorizontal: 12,
    borderRadius: 10,
    backgroundColor: 'rgba(60, 60, 67, 0.12)',
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
  imageProgressText: {
    color: '#fff',
    marginLeft: 10,
    fontWeight: '500',
    fontSize: 15,
  },
  resultContainer: {
    marginTop: 16,
    marginBottom: 100, // Increase the bottom margin for more space
    padding: 0,
    borderRadius: 12,
    overflow: 'hidden',
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
  // NSGS Processing styles
  nsgsProcessingContainer: {
    width: '100%',
    padding: 12,
  },
  nsgsHeader: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 12,
    textAlign: 'center',
  },
  nsgsStatusText: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.8,
    textAlign: 'center',
    marginBottom: 16,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  nsgsMetricContainer: {
    marginBottom: 12,
  },
  nsgsMetricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  nsgsMetricTitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
    fontWeight: '500',
  },
  nsgsMetricValue: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '700',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  nsgsProgressBarContainer: {
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  nsgsProgressBar: {
    height: '100%',
    backgroundColor: '#4A90E2',
  },
  nsgsProcessingTime: {
    fontSize: 15,
    color: '#fff',
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 16,
  },
  triangleLoadingContainer: {
    width: '100%',
    padding: 12,
    alignItems: 'center',
  },
  triangleLoadingTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 12,
    textAlign: 'center',
  },
  triangleLoadingStatus: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.8,
    textAlign: 'center',
    marginBottom: 16,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  triangleWrapper: {
    width: 280,
    height: 320, // Taller container for the D-shape
    position: 'relative',
    marginBottom: 16,
  },
  triangleLoadingTimer: {
    fontSize: 15,
    color: '#fff',
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 8,
  },
  nsgsMetricsContainer: {
    marginTop: 20,
    padding: 15,
    borderRadius: 12,
    backgroundColor: 'rgba(0,0,0,0.1)',
    maxHeight: 250,
  },
  nsgsMetricsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
    textAlign: 'center',
  },
  nsgsMetricsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  nsgsMetricBlock: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    marginHorizontal: 4,
  },
  nsgsMetricLabel: {
    fontSize: 13,
    marginBottom: 4,
  },
  nsgsMetricsTimeContainer: {
    alignItems: 'center',
    marginTop: 8,
    padding: 10,
    borderRadius: 8,
  },
  nsgsMetricsTimeLabel: {
    fontSize: 13,
    marginBottom: 4,
  },
  nsgsMetricsTimeValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#6A35D9',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  resultsContainer: {
    marginTop: 16,
    marginBottom: 100, // Increase the bottom margin for more space
    padding: 0,
    borderRadius: 12,
    overflow: 'hidden',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginLeft: 10,
    fontWeight: '600',
    fontSize: 15,
  },
  progressText: {
    marginLeft: 10,
    fontWeight: '500',
    fontSize: 15,
  },
  resultImageContainer: {
    width: '100%',
    borderRadius: BORDER_RADIUS,
    overflow: 'hidden',
    ...CARD_SHADOW,
    position: 'relative',
    marginBottom: 15,
  },
  resultImage: {
    width: '100%',
    height: 300,
    borderRadius: BORDER_RADIUS,
    marginBottom: 12,
  },
  debugInfoContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  debugInfoContainerWithSlideshow: {
    bottom: 70, // Move up to allow space for slideshow controls
  },
  debugInfoText: {
    fontSize: 13,
    color: '#fff',
  },
  reconnectButton: {
    padding: 16,
    borderRadius: 10,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  reconnectButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
  },
  refreshButton: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 30,
    height: 30,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  refreshButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  slideshowControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 10,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
  },
  slideshowButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(106, 53, 217, 0.8)',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 10,
  },
  slideshowButtonDisabled: {
    backgroundColor: 'rgba(128, 128, 128, 0.4)',
  },
  slideshowButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  slideshowCounter: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
  },
  slideshowCounterText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  dotIndicatorContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 4,
  },
  dotIndicator: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginHorizontal: 3,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
  dotIndicatorActive: {
    backgroundColor: '#fff',
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  imageTitleContainer: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    zIndex: 5,
    textAlign: 'center',
  },
  imageTitleText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  standalonensgsMetricsContainer: {
    width: '100%',
    marginTop: 20,
    marginBottom: 200,
    padding: 15,
    borderRadius: BORDER_RADIUS,
    ...CARD_SHADOW,
  },
  nsgsMetricsHeaderText: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    textAlign: 'center',
  },
  nsgsMetricsScrollView: {
    maxHeight: 250,
  },
  nsgsLogsContainer: {
    marginTop: 16,
    width: '100%',
  },
  nsgsLogsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    textAlign: 'center',
  },
  nsgsLogsScrollView: {
    maxHeight: 1000, // Significantly increase the fixed height to show much more content
    minHeight: 800, // Increase minimum height
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 10,
    marginBottom: 40, // Add bottom margin
  },
  nsgsLogsText: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 12,
    lineHeight: 18,
  },
  debugRefreshButton: {
    padding: 10,
    borderRadius: 10,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  debugRefreshText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
  },
  logStatusContainer: {
    marginBottom: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  logStatusText: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 4,
  },
  standaloneLogsContainer: {
    width: '100%',
    marginTop: 30,
    marginBottom: 200,
    padding: 20,
    paddingBottom: 30,
    borderRadius: BORDER_RADIUS,
    ...CARD_SHADOW,
    borderWidth: 2,
    borderColor: '#6A35D9', // Use a distinctive color for the logs container
  },
});