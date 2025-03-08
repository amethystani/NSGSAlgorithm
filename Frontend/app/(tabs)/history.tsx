import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Clock } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import Animated, { 
  FadeIn,
  useAnimatedStyle, 
  useSharedValue, 
  withTiming 
} from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';

// Define types for HistoryItem props
interface HistoryItemProps {
  label: string;
  children: React.ReactNode;
  last?: boolean;
}

// iOS 18 inspired history item component
const HistoryItem = ({ label, children, last = false }: HistoryItemProps) => {
  const { theme } = useTheme();
  const opacity = useSharedValue(1);
  
  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      backgroundColor: opacity.value < 1 ? theme.placeholder : theme.card,
    };
  });
  
  return (
    <Pressable
      onPressIn={() => {
        Haptics.selectionAsync();
        opacity.value = withTiming(0.8, { duration: 100 });
      }}
      onPressOut={() => {
        opacity.value = withTiming(1, { duration: 200 });
      }}
      style={({ pressed }) => [
        pressed && { backgroundColor: theme.placeholder }
      ]}
    >
      <Animated.View style={[
        styles.historyItem, 
        !last && [styles.historyItemBorder, { borderBottomColor: theme.border }],
        { backgroundColor: theme.card },
        animatedStyle
      ]}>
        <Text style={[styles.historyItemLabel, { color: theme.text }]}>{label}</Text>
        <View style={styles.historyItemDetail}>
          {children}
        </View>
      </Animated.View>
    </Pressable>
  );
};

export default function HistoryScreen() {
  const { theme, isDarkMode } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);  // This would normally be populated with actual history data
  
  // Use a shorter timeout to make UI appear faster
  useEffect(() => {
    const timer = setTimeout(() => {
      setMounted(true);
    }, 10);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Text style={[styles.title, { color: theme.text }]}>History</Text>
        </View>

        {historyItems.length > 0 ? (
          <Animated.View
            entering={mounted ? FadeIn.delay(100).duration(300) : undefined}
            style={styles.section}>
            <Text style={[styles.sectionTitle, { color: theme.text }]}>Recent Detections</Text>
            <View style={[styles.card, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
              {/* This would render actual history items when available */}
            </View>
          </Animated.View>
        ) : (
          <Animated.View 
            entering={mounted ? FadeIn.delay(150).duration(300) : undefined}
            style={styles.emptyStateContainer}>
            <View style={[styles.emptyState, { backgroundColor: theme.card, shadowColor: isDarkMode ? 'transparent' : '#000' }]}>
              <Clock size={48} color={theme.textSecondary} style={styles.emptyStateIcon} />
              <Text style={[styles.emptyStateText, { color: theme.text }]}>No Detection History</Text>
              <Text style={[styles.emptyStateSubtext, { color: theme.textSecondary }]}>
                Your detection history will appear here
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
  historyItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  historyItemBorder: {
    borderBottomWidth: 0.5,
  },
  historyItemLabel: {
    fontSize: 17,
    fontWeight: '500',
    flex: 1,
  },
  historyItemDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    minWidth: 60,
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