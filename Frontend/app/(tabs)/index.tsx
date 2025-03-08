import { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, Upload, Image as ImageIcon } from 'lucide-react-native';
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

const AnimatedBlurView = Animated.createAnimatedComponent(BlurView);

export default function DetectScreen() {
  const [selectedModel, setSelectedModel] = useState('yolov8m.pt');
  const [processing, setProcessing] = useState(false);
  const { theme, isDarkMode } = useTheme();

  // Function to trigger haptic feedback on model selection
  const handleModelSelect = useCallback((model: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setSelectedModel(model);
  }, []);

  // Function for camera capture with haptic feedback
  const takePhoto = useCallback(async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    const result = await ImagePicker.launchCameraAsync({
      quality: 1,
    });

    if (!result.canceled) {
      setProcessing(true);
      // Trigger success haptic when processing starts
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => {
        setProcessing(false);
        // Trigger another haptic when processing completes
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }, 2000);
    }
  }, []);

  // Function for picking image with haptic feedback
  const pickImage = useCallback(async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: true,
      quality: 1,
    });

    if (!result.canceled) {
      setProcessing(true);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => {
        setProcessing(false);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }, 2000);
    }
  }, []);

  // Function for upload with haptic feedback
  const handleUpload = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // Upload logic would go here
  }, []);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollViewContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={[styles.title, { color: theme.text }]}>Object Detection</Text>
        </View>

        {/* Model Selection */}
        <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Model Type</Text>
          <View style={[styles.segmentedControlContainer, { backgroundColor: theme.card }]}>
            <View style={[styles.segmentedControl, { backgroundColor: isDarkMode ? '#2C2C2E' : '#F2F2F7' }]}>
              {['yolov8m.pt', 'yolov8m-seg.pt'].map((model) => {
                const isActive = selectedModel === model;
                return (
                  <Animated.View
                    key={model}
                    style={[
                      styles.segmentWrapper,
                      model === 'yolov8m.pt' && styles.segmentWrapperLeft,
                      model === 'yolov8m-seg.pt' && styles.segmentWrapperRight,
                    ]}
                  >
                    <TouchableOpacity
                      style={[
                        styles.segment,
                        isActive && [styles.segmentActive, { backgroundColor: theme.card }],
                      ]}
                      onPress={() => handleModelSelect(model)}
                      activeOpacity={0.9}
                    >
                      <Text style={[
                        styles.segmentLabel,
                        { color: theme.textSecondary },
                        isActive && [styles.segmentLabelActive, { color: theme.text }]
                      ]}>
                        {model.includes('seg') ? 'Segmentation' : 'Detection'}
                      </Text>
                    </TouchableOpacity>
                  </Animated.View>
                );
              })}
            </View>
          </View>
        </View>

        {/* Input Options */}
        <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Input Source</Text>
          <View style={styles.inputGrid}>
            {[
              { icon: Camera, label: 'Camera', action: takePhoto, color: '#34C759' },
              { icon: ImageIcon, label: 'Gallery', action: pickImage, color: '#0A84FF' },
              { icon: Upload, label: 'Upload', action: handleUpload, color: '#FF453A' }
            ].map(({ icon: Icon, label, action, color }, index) => (
              <TouchableOpacity
                key={label}
                style={[styles.inputCard, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}
                onPress={action}
                activeOpacity={0.7}
              >
                <View style={[
                  styles.iconWrapper,
                  { backgroundColor: `${color}16` } // 10% opacity version of the color
                ]}>
                  <Icon size={28} color={color} strokeWidth={2.5} />
                </View>
                <Text style={[styles.inputLabel, { color: theme.text }]}>{label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Results Section */}
        <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Analysis Results</Text>
          <View style={styles.resultsPlaceholder}>
            <View style={[styles.placeholderIconContainer, { backgroundColor: isDarkMode ? '#2C2C2E' : '#F2F2F7' }]}>
              <ImageIcon size={36} color={theme.textSecondary} strokeWidth={1.5} />
            </View>
            <Text style={[styles.placeholderTitle, { color: theme.text }]}>
              No Results Yet
            </Text>
            <Text style={[styles.placeholderText, { color: theme.textSecondary }]}>
              Select an image to begin object detection
            </Text>
          </View>
        </View>
      </ScrollView>

      {/* Processing Overlay */}
      {processing && (
        <AnimatedBlurView
          entering={FadeIn.duration(300)}
          exiting={FadeOut.duration(300)}
          style={styles.processingOverlay}
          intensity={15}
          tint={isDarkMode ? "dark" : "light"}
        >
          <Animated.View
            entering={SlideInDown.springify().damping(15)}
            style={[styles.processingContent, 
            { backgroundColor: isDarkMode ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.9)', 
              shadowColor: isDarkMode ? 'transparent' : '#000' }]}
          >
            <Text style={[styles.processingTitle, { color: theme.text }]}>Analyzing Image</Text>
            <View style={styles.progressContainer}>
              <View style={[styles.progressBar, { backgroundColor: isDarkMode ? '#2C2C2E' : '#E5E5EA' }]}>
                <Animated.View 
                  style={[styles.progressBarFill]}
                />
              </View>
              <Text style={[styles.progressText, { color: theme.textSecondary }]}>60% Complete</Text>
            </View>
          </Animated.View>
        </AnimatedBlurView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollViewContent: {
    paddingBottom: 32,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 10,
  },
  title: {
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  card: {
    borderRadius: 16,
    marginHorizontal: 20,
    marginBottom: 16,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 2,
    overflow: 'hidden',
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginVertical: 16,
    marginHorizontal: 16,
  },
  segmentedControlContainer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  segmentedControl: {
    flexDirection: 'row',
    borderRadius: 10,
    overflow: 'hidden',
    padding: 2,
    height: 48,
  },
  segmentWrapper: {
    flex: 1,
  },
  segmentWrapperLeft: {
    borderTopLeftRadius: 8,
    borderBottomLeftRadius: 8,
  },
  segmentWrapperRight: {
    borderTopRightRadius: 8,
    borderBottomRightRadius: 8,
  },
  segment: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
  },
  segmentActive: {
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  segmentLabel: {
    fontSize: 16,
    fontWeight: '500',
  },
  segmentLabelActive: {
    fontWeight: '600',
  },
  inputGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 20,
    paddingTop: 8,
  },
  inputCard: {
    borderRadius: 14,
    padding: 12,
    width: '30%',
    alignItems: 'center',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  iconWrapper: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  inputLabel: {
    fontSize: 15,
    fontWeight: '600',
  },
  resultsPlaceholder: {
    paddingVertical: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderIconContainer: {
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  placeholderTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 8,
  },
  placeholderText: {
    fontSize: 15,
    textAlign: 'center',
    maxWidth: '80%',
  },
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  processingContent: {
    borderRadius: 16,
    padding: 24,
    width: '90%',
    maxWidth: 400,
    alignItems: 'center',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 10,
  },
  processingTitle: {
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 24,
    textAlign: 'center',
  },
  progressContainer: {
    width: '100%',
  },
  progressBar: {
    height: 8,
    width: '100%',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    width: '60%',
    backgroundColor: '#0A84FF',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 15,
    textAlign: 'center',
    marginTop: 12,
  },
});