import { Platform } from 'react-native';
import * as Haptics from 'expo-haptics';

/**
 * Trigger haptic feedback with impact style only on native platforms
 * @param style Impact feedback style
 */
export const triggerHaptic = (style: Haptics.ImpactFeedbackStyle) => {
  if (Platform.OS !== 'web') {
    Haptics.impactAsync(style);
  }
};

/**
 * Trigger haptic notification feedback only on native platforms
 * @param type Notification feedback type
 */
export const triggerHapticNotification = (type: Haptics.NotificationFeedbackType) => {
  if (Platform.OS !== 'web') {
    Haptics.notificationAsync(type);
  }
}; 