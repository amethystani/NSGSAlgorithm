import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Image, TouchableOpacity, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Image as ImageIcon } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import Animated, { 
  FadeIn,
  useAnimatedStyle, 
  useSharedValue, 
  withTiming 
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';

const DEMO_IMAGES = [
  'https://images.unsplash.com/photo-1517849845537-4d257902454a',
  'https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9',
  'https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f',
];

// iOS 18 inspired image item component
const GalleryImage = ({ url, index }: { url: string, index: number }) => {
  const { theme } = useTheme();
  const opacity = useSharedValue(1);
  const scale = useSharedValue(1);
  
  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [{ scale: scale.value }],
    };
  });
  
  return (
    <Pressable
      onPressIn={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        opacity.value = withTiming(0.9, { duration: 100 });
        scale.value = withTiming(0.98, { duration: 100 });
      }}
      onPressOut={() => {
        opacity.value = withTiming(1, { duration: 200 });
        scale.value = withTiming(1, { duration: 200 });
      }}
    >
      <Animated.View 
        style={[
          styles.imageContainer,
          { backgroundColor: theme.card, shadowColor: theme.text === '#FFFFFF' ? 'transparent' : '#000' },
          animatedStyle
        ]}
      >
        <Image
          source={{ uri: `${url}?w=400&fit=crop` }}
          style={styles.image}
        />
        <View style={[styles.imageOverlay, { backgroundColor: theme.card }]}>
          <Text style={[styles.detectionCount, { color: theme.text }]}>5 objects</Text>
        </View>
      </Animated.View>
    </Pressable>
  );
};

export default function GalleryScreen() {
  const { theme, isDarkMode } = useTheme();
  const [mounted, setMounted] = useState(false);
  
  // Use a shorter timeout to make UI appear faster
  useEffect(() => {
    const timer = setTimeout(() => {
      setMounted(true);
    }, 10);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      <ScrollView 
        contentContainerStyle={styles.scrollContent} 
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <Text style={[styles.title, { color: theme.text }]}>Gallery</Text>
        </View>
        
        {DEMO_IMAGES.length > 0 ? (
          <Animated.View
            entering={mounted ? FadeIn.delay(100).duration(300) : undefined}
            style={styles.section}
          >
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Processed Images</Text>
            <View style={styles.grid}>
              {DEMO_IMAGES.map((url, index) => (
                <Animated.View
                  key={index}
                  entering={mounted ? FadeIn.delay(150 + index * 50).duration(300) : undefined}
                >
                  <GalleryImage url={url} index={index} />
                </Animated.View>
              ))}
            </View>
          </Animated.View>
        ) : (
          <Animated.View 
            entering={mounted ? FadeIn.delay(150).duration(300) : undefined}
            style={styles.emptyStateContainer}
          >
            <View style={[styles.emptyState, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
              <ImageIcon size={48} color={theme.textSecondary} style={styles.emptyStateIcon} />
              <Text style={[styles.emptyStateText, { color: theme.text }]}>
                No processed images yet
              </Text>
              <Text style={[styles.emptyStateSubtext, { color: theme.textSecondary }]}>
                Images will appear here after processing
              </Text>
            </View>
          </Animated.View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 40,
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
    marginTop: 10,
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 8,
    marginLeft: 10,
  },
  grid: {
    flexDirection: 'column',
    gap: 16,
  },
  imageContainer: {
    width: '100%',
    borderRadius: 16,
    overflow: 'hidden',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 2,
  },
  image: {
    width: '100%',
    aspectRatio: 16 / 9,
  },
  imageOverlay: {
    padding: 16,
  },
  detectionCount: {
    fontSize: 17,
    fontWeight: '500',
  },
  emptyStateContainer: {
    marginTop: 60,
    paddingHorizontal: 20,
  },
  emptyState: {
    borderRadius: 16,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 2,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
    paddingHorizontal: 20,
  },
  emptyStateIcon: {
    marginBottom: 16,
  },
  emptyStateText: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 15,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
});