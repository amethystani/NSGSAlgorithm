import React, { useEffect } from 'react';
import { Tabs } from 'expo-router';
import { Camera, Settings, Image as ImageIcon, History } from 'lucide-react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Platform, StyleSheet, Pressable, View, Text } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { 
  useAnimatedStyle, 
  useSharedValue, 
  withTiming
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';
import { BlurView } from 'expo-blur';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';

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

// Custom tab bar that incorporates the blur effect
const CustomTabBar: React.FC<BottomTabBarProps> = ({ state, descriptors, navigation }) => {
  const insets = useSafeAreaInsets();
  const { theme, isDarkMode } = useTheme();
  
  return (
    <BlurView
      intensity={isDarkMode ? 40 : 55}
      tint={isDarkMode ? "dark" : "light"}
      style={[
        styles.tabBarContainer, 
        { 
          height: 62 + Math.max(insets.bottom, 10),
          paddingBottom: Math.max(insets.bottom, 10),
        }
      ]}
    >
      {state.routes.map((route, index) => {
        const { options } = descriptors[route.key];
        const isFocused = state.index === index;
        
        // Handle label - can be string or function
        let labelText: string;
        if (typeof options.tabBarLabel === 'string') {
          labelText = options.tabBarLabel;
        } else if (typeof options.title === 'string') {
          labelText = options.title;
        } else {
          labelText = route.name;
        }
        
        const onPress = () => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          const event = navigation.emit({
            type: 'tabPress',
            target: route.key,
            canPreventDefault: true,
          });

          if (!isFocused && !event.defaultPrevented) {
            navigation.navigate(route.name);
          }
        };

        return (
          <Pressable
            key={route.key}
            accessibilityRole="button"
            accessibilityState={isFocused ? { selected: true } : {}}
            onPress={onPress}
            style={styles.tabButton}
          >
            {options.tabBarIcon && 
              options.tabBarIcon({
                focused: isFocused,
                color: isFocused ? theme.primary : theme.text,
                size: 24
              })
            }
            <Text
              style={[
                styles.tabBarLabel,
                { 
                  color: isFocused ? theme.primary : theme.text,
                }
              ]}
            >
              {labelText}
            </Text>
          </Pressable>
        );
      })}
    </BlurView>
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
      tabBar={props => <CustomTabBar {...props} />}
      screenOptions={{
        headerShown: false,
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Detect',
          tabBarIcon: ({ color, size, focused }) => (
            <AnimatedTabBarIcon icon={Camera} color={color} size={size} focused={focused} />
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
        }}
      />
      <Tabs.Screen
        name="gallery"
        options={{
          title: 'Gallery',
          tabBarIcon: ({ color, size, focused }) => (
            <AnimatedTabBarIcon icon={ImageIcon} color={color} size={size} focused={focused} />
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
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBarContainer: {
    flexDirection: 'row',
    borderTopWidth: 0,
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.06,
    shadowRadius: 10,
    elevation: 10,
  },
  tabButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
  },
  tabBarLabel: {
    fontSize: 11,
    fontWeight: '600',
    marginTop: 4,
    letterSpacing: -0.2,
  }
});