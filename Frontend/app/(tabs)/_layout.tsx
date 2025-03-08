import React, { useEffect } from 'react';
import { Tabs } from 'expo-router';
import { Camera, Settings, Image as ImageIcon, History } from 'lucide-react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Platform, StyleSheet, Pressable, View } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { 
  useAnimatedStyle, 
  useSharedValue, 
  withTiming
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';

// Custom animated tab bar icon component for iOS 18 style
const AnimatedTabBarIcon = ({ 
  icon: Icon, 
  color, 
  size, 
  focused 
}: { 
  icon: any, 
  color: string, 
  size: number, 
  focused: boolean 
}) => {
  const scale = useSharedValue(1);
  
  useEffect(() => {
    scale.value = withTiming(focused ? 1.1 : 1, { duration: 200 });
  }, [focused]);
  
  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }]
    };
  });
  
  return (
    <Animated.View style={animatedStyle}>
      <Icon size={size} color={color} strokeWidth={focused ? 2.5 : 2} />
    </Animated.View>
  );
};

export default function TabLayout() {
  const insets = useSafeAreaInsets();
  const { theme, isDarkMode } = useTheme();
  
  // Function for haptic feedback
  const handleTabPress = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };
  
  return (
    <Tabs
      screenOptions={{
        tabBarStyle: {
          backgroundColor: isDarkMode 
            ? theme.card
            : 'rgba(255, 255, 255, 0.98)',
          borderTopWidth: 0,
          shadowColor: isDarkMode ? 'transparent' : '#000',
          shadowOffset: { width: 0, height: -2 },
          shadowOpacity: 0.06,
          shadowRadius: 10,
          elevation: 10,
          height: 62 + Math.max(insets.bottom, 10),
          paddingBottom: Math.max(insets.bottom, 10),
          paddingTop: 8,
          position: 'absolute',
        },
        tabBarActiveTintColor: theme.primary,
        tabBarInactiveTintColor: theme.text,
        headerShown: false,
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: '600',
          paddingTop: 4,
          letterSpacing: -0.2,
        },
        tabBarIconStyle: {
          marginBottom: -4,
        },
        tabBarItemStyle: {
          paddingVertical: 6,
        },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Detect',
          tabBarIcon: ({ color, size, focused }) => (
            <AnimatedTabBarIcon icon={Camera} color={color} size={size} focused={focused} />
          ),
          tabBarButton: (props) => (
            <Pressable
              {...props}
              onPress={(e) => {
                handleTabPress();
                // @ts-ignore
                props.onPress && props.onPress(e);
              }}
              style={({ pressed }) => [
                { opacity: pressed ? 0.8 : 1 },
              ]}
            >
              <View style={styles.tabContainer}>
                {/* @ts-ignore */}
                <View style={props.style}>
                  {/* @ts-ignore */}
                  {props.children}
                </View>
              </View>
            </Pressable>
          ),
        }}
      />
      <Tabs.Screen
        name="gallery"
        options={{
          title: 'Gallery',
          tabBarIcon: ({ color, size, focused }) => (
            <AnimatedTabBarIcon icon={ImageIcon} color={color} size={size} focused={focused} />
          ),
          tabBarButton: (props) => (
            <Pressable
              {...props}
              onPress={(e) => {
                handleTabPress();
                // @ts-ignore
                props.onPress && props.onPress(e);
              }}
              style={({ pressed }) => [
                { opacity: pressed ? 0.8 : 1 },
              ]}
            >
              <View style={styles.tabContainer}>
                {/* @ts-ignore */}
                <View style={props.style}>
                  {/* @ts-ignore */}
                  {props.children}
                </View>
              </View>
            </Pressable>
          ),
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: 'History',
          tabBarIcon: ({ color, size, focused }) => (
            <AnimatedTabBarIcon icon={History} color={color} size={size} focused={focused} />
          ),
          tabBarButton: (props) => (
            <Pressable
              {...props}
              onPress={(e) => {
                handleTabPress();
                // @ts-ignore
                props.onPress && props.onPress(e);
              }}
              style={({ pressed }) => [
                { opacity: pressed ? 0.8 : 1 },
              ]}
            >
              <View style={styles.tabContainer}>
                {/* @ts-ignore */}
                <View style={props.style}>
                  {/* @ts-ignore */}
                  {props.children}
                </View>
              </View>
            </Pressable>
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color, size, focused }) => (
            <AnimatedTabBarIcon icon={Settings} color={color} size={size} focused={focused} />
          ),
          tabBarButton: (props) => (
            <Pressable
              {...props}
              onPress={(e) => {
                handleTabPress();
                // @ts-ignore
                props.onPress && props.onPress(e);
              }}
              style={({ pressed }) => [
                { opacity: pressed ? 0.8 : 1 },
              ]}
            >
              <View style={styles.tabContainer}>
                {/* @ts-ignore */}
                <View style={props.style}>
                  {/* @ts-ignore */}
                  {props.children}
                </View>
              </View>
            </Pressable>
          ),
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
  }
});