import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Switch, Pressable, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import { Moon, Sun } from 'lucide-react-native';
import Animated, { 
  SlideInRight, 
  FadeIn,
  useAnimatedStyle, 
  useSharedValue, 
  withTiming 
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';

// Define types for SettingItem props
interface SettingItemProps {
  label: string;
  children: React.ReactNode;
  last?: boolean;
}

// iOS 18 inspired setting item component
const SettingItem = ({ label, children, last = false }: SettingItemProps) => {
  const { theme } = useTheme();
  const opacity = useSharedValue(1);  // Increase default opacity for better visibility
  
  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      backgroundColor: opacity.value < 1 ? theme.placeholder : theme.card, // Add background color change on press
    };
  });
  
  return (
    <Pressable
      onPressIn={() => {
        Haptics.selectionAsync(); // Add haptic feedback on press
        opacity.value = withTiming(0.8, { duration: 100 });
      }}
      onPressOut={() => {
        opacity.value = withTiming(1, { duration: 200 });
      }}
      style={({ pressed }) => [
        pressed && { backgroundColor: theme.placeholder } // Fallback for when animation might not work
      ]}
    >
      <Animated.View style={[
        styles.setting, 
        !last && [styles.settingBorder, { borderBottomColor: theme.border }],
        { backgroundColor: theme.card },
        animatedStyle
      ]}>
        <Text style={[styles.settingLabel, { color: theme.text }]}>{label}</Text>
        <View style={styles.settingControl}>
          {children}
        </View>
      </Animated.View>
    </Pressable>
  );
};

export default function SettingsScreen() {
  const { theme, isDarkMode, toggleTheme, isSystemTheme, setIsSystemTheme } = useTheme();
  const [useGPU, setUseGPU] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.4);
  const [iouThreshold, setIouThreshold] = useState(0.4);
  const [maskThreshold, setMaskThreshold] = useState(0.5);
  const [mounted, setMounted] = useState(false);

  // Handler for toggle switch with haptic feedback
  const handleToggleChange = (newValue: boolean) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setUseGPU(newValue);
  };

  // Handler for theme toggle
  const handleThemeToggle = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    toggleTheme();
  };

  // Handler for system theme toggle
  const handleSystemThemeToggle = (value: boolean) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setIsSystemTheme(value);
  };

  // Use a shorter timeout to make UI appear faster
  useEffect(() => {
    // Make the UI immediately visible with a very short delay for animations
    const timer = setTimeout(() => {
      setMounted(true);
    }, 10);
    
    return () => clearTimeout(timer);
  }, []);

  // Create a slider component for the threshold settings
  const ThresholdValue = ({ value }: { value: number }) => (
    <Text style={[styles.settingValue, { color: theme.textSecondary }]}>{value.toFixed(2)}</Text>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <Text style={[styles.title, { color: theme.text }]}>Settings</Text>
        </View>
        
        <Animated.View
          entering={mounted ? FadeIn.delay(50).duration(300) : undefined}
          style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Appearance</Text>
          <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
            <SettingItem label="Use System Setting">
              <View style={styles.settingInfo}>
                <Text style={[styles.settingHint, { color: theme.textSecondary }]}>
                  {isSystemTheme ? "On" : "Off"}
                </Text>
                <Switch
                  value={isSystemTheme}
                  onValueChange={handleSystemThemeToggle}
                  trackColor={{ false: '#D1D1D6', true: theme.success }}
                  thumbColor={'#FFFFFF'}
                  ios_backgroundColor="#D1D1D6"
                />
              </View>
            </SettingItem>
            <SettingItem label="Dark Mode" last>
              <View style={styles.themeToggle}>
                {isDarkMode ? (
                  <Moon size={22} color={theme.primary} />
                ) : (
                  <Sun size={22} color={theme.primary} />
                )}
                <Switch
                  value={isDarkMode}
                  onValueChange={handleThemeToggle}
                  disabled={isSystemTheme}
                  trackColor={{ false: '#D1D1D6', true: theme.primary }}
                  thumbColor={'#FFFFFF'}
                  ios_backgroundColor="#D1D1D6"
                  style={{ opacity: isSystemTheme ? 0.5 : 1, marginLeft: 8 }}
                />
              </View>
            </SettingItem>
          </View>
          {isSystemTheme && (
            <Text style={[styles.helpText, { color: theme.textSecondary }]}>
              Using your device's appearance settings
            </Text>
          )}
        </Animated.View>

        <Animated.View
          entering={mounted ? FadeIn.delay(100).duration(300) : undefined}
          style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Processing</Text>
          <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
            <SettingItem label="Use GPU" last>
              <Switch
                value={useGPU}
                onValueChange={handleToggleChange}
                trackColor={{ false: '#D1D1D6', true: theme.success }}
                thumbColor={'#FFFFFF'}
                ios_backgroundColor="#D1D1D6"
              />
            </SettingItem>
          </View>
        </Animated.View>

        <Animated.View
          entering={mounted ? FadeIn.delay(150).duration(300) : undefined}
          style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Detection Settings</Text>
          <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
            <SettingItem label="Confidence Threshold">
              <ThresholdValue value={confidenceThreshold} />
            </SettingItem>
            <SettingItem label="IoU Threshold">
              <ThresholdValue value={iouThreshold} />
            </SettingItem>
            <SettingItem label="Mask Threshold" last>
              <ThresholdValue value={maskThreshold} />
            </SettingItem>
          </View>
        </Animated.View>

        <Animated.View
          entering={mounted ? FadeIn.delay(200).duration(300) : undefined}
          style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>About</Text>
          <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
            <SettingItem label="Version" last>
              <Text style={[styles.settingValue, { color: theme.textSecondary }]}>1.0.0</Text>
            </SettingItem>
          </View>
        </Animated.View>
      </ScrollView>
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
  scrollContent: {
    paddingBottom: 30,
  },
  header: {
    padding: 20,
    paddingTop: 10,
  },
  title: {
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  section: {
    marginTop: 20,
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 8,
    marginLeft: 10,
  },
  card: {
    borderRadius: 16,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 2,
    overflow: 'hidden',
  },
  setting: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  settingBorder: {
    borderBottomWidth: 0.5,
  },
  settingLabel: {
    fontSize: 17,
    fontWeight: '500',
    flex: 1, // Allow the label to take available space
  },
  settingValue: {
    fontSize: 17,
    fontWeight: '400',
  },
  settingControl: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    minWidth: 60, // Ensure there's space for controls
  },
  themeToggle: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingHint: {
    fontSize: 16,
    marginRight: 8,
  },
  helpText: {
    fontSize: 13,
    color: '#8E8E93',
    marginTop: 6,
    marginLeft: 10,
    fontStyle: 'italic',
  }
});