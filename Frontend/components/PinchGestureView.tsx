import React from 'react';
import { StyleSheet, View } from 'react-native';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
  runOnJS
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';

interface PinchGestureViewProps {
  children: React.ReactNode;
  maxScale?: number;
  minScale?: number;
  onPinchStart?: () => void;
  onPinchEnd?: () => void;
  onDoubleTap?: () => void;
}

export const PinchGestureView: React.FC<PinchGestureViewProps> = ({
  children,
  maxScale = 5,
  minScale = 1,
  onPinchStart,
  onPinchEnd,
  onDoubleTap,
}) => {
  // Initialize shared values for transformations
  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const lastScale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const lastTranslateX = useSharedValue(0);
  const lastTranslateY = useSharedValue(0);

  const triggerHapticFeedback = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(err => {
      console.log('Haptic feedback error:', err);
    });
  };

  const triggerMediumHapticFeedback = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(err => {
      console.log('Haptic feedback error:', err);
    });
  };

  // Define the pinch gesture
  const pinchGesture = Gesture.Pinch()
    .onStart(() => {
      if (onPinchStart) {
        runOnJS(onPinchStart)();
      }
    })
    .onUpdate((event) => {
      // Calculate new scale value based on the pinch gesture
      let newScale = savedScale.value * event.scale;
      
      // Prevent zooming out beyond the original size (1.0)
      // This is the key change to prevent the image from going out of canvas
      if (newScale < 1.0) {
        newScale = 1.0;
      }
      
      // Update scale
      scale.value = newScale;
      
      // Provide haptic feedback when scale changes significantly
      if (Math.abs(scale.value - lastScale.value) > 0.5) {
        runOnJS(triggerHapticFeedback)();
        lastScale.value = scale.value;
      }
    })
    .onEnd(() => {
      // Apply constraints to the scale value
      // Always ensure we don't go below 1.0 (original size)
      if (scale.value < minScale) {
        scale.value = withTiming(minScale);
        savedScale.value = minScale;
      } else if (scale.value > maxScale) {
        scale.value = withTiming(maxScale);
        savedScale.value = maxScale;
      } else {
        savedScale.value = scale.value;
      }
      
      if (onPinchEnd) {
        runOnJS(onPinchEnd)();
      }
    });

  // Define the pan gesture for moving the content when zoomed in
  const panGesture = Gesture.Pan()
    .onUpdate((event) => {
      // Only allow panning when zoomed in
      if (scale.value > 1) {
        // Calculate max allowed translation based on current scale
        // This helps keep the image within visible bounds when panning
        const maxTranslateX = (scale.value - 1) * 150;
        const maxTranslateY = (scale.value - 1) * 150;
        
        // Apply translation with constraints
        let newTranslateX = lastTranslateX.value + event.translationX;
        let newTranslateY = lastTranslateY.value + event.translationY;
        
        // Constrain the translation to keep image in view
        newTranslateX = Math.min(Math.max(newTranslateX, -maxTranslateX), maxTranslateX);
        newTranslateY = Math.min(Math.max(newTranslateY, -maxTranslateY), maxTranslateY);
        
        translateX.value = newTranslateX;
        translateY.value = newTranslateY;
      }
    })
    .onEnd(() => {
      lastTranslateX.value = translateX.value;
      lastTranslateY.value = translateY.value;
    });

  // Define double tap gesture to reset zoom
  const doubleTapGesture = Gesture.Tap()
    .numberOfTaps(2)
    .maxDuration(250)
    .onEnd(() => {
      // Reset transformations
      scale.value = withTiming(1);
      savedScale.value = 1;
      lastScale.value = 1;
      translateX.value = withTiming(0);
      translateY.value = withTiming(0);
      lastTranslateX.value = 0;
      lastTranslateY.value = 0;
      
      // Trigger haptic feedback for the reset action
      runOnJS(triggerMediumHapticFeedback)();
      
      if (onDoubleTap) {
        runOnJS(onDoubleTap)();
      }
    });

  // Combine gestures with proper priority/composing
  const combinedGesture = Gesture.Race(
    doubleTapGesture,
    Gesture.Simultaneous(pinchGesture, panGesture)
  );

  // Define animated styles for transformations
  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { scale: scale.value }
      ]
    };
  });

  return (
    <View style={styles.container}>
      <GestureDetector gesture={combinedGesture}>
        <Animated.View style={[styles.content, animatedStyle]}>
          {children}
        </Animated.View>
      </GestureDetector>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    overflow: 'hidden',
  },
  content: {
    flex: 1,
    width: '100%',
    height: '100%',
  }
}); 