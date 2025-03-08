import React, { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useFrameworkReady } from '@/hooks/useFrameworkReady';
import { View } from 'react-native';
import Animated, { FadeIn } from 'react-native-reanimated';
import { ThemeProvider, useTheme } from '@/context/ThemeContext';

// Wrap the app content with theme context
function AppContent() {
  useFrameworkReady();
  const { theme, isDarkMode } = useTheme();

  return (
    <Animated.View 
      style={{ flex: 1, backgroundColor: theme.background }}
      entering={FadeIn.duration(300)}
    >
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: theme.background },
          animation: 'fade',
          animationDuration: 200,
        }}>
        <Stack.Screen 
          name="(tabs)" 
          options={{ 
            animation: 'fade',
            gestureEnabled: false,
          }} 
        />
        <Stack.Screen name="+not-found" options={{ animation: 'slide_from_right' }} />
      </Stack>
      <StatusBar style={isDarkMode ? "light" : "dark"} animated={true} />
    </Animated.View>
  );
}

export default function RootLayout() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
