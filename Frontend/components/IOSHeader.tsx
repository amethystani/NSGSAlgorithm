import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';
import { useNavigation, useRouter } from 'expo-router';
import { ArrowLeft } from 'lucide-react-native';
import Animated, { FadeIn } from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';

interface IOSHeaderProps {
  title: string;
  showBackButton?: boolean;
  rightComponent?: React.ReactNode;
  largeTitleStyle?: boolean;
}

export const IOSHeader = ({ 
  title, 
  showBackButton = false, 
  rightComponent = null,
  largeTitleStyle = true,
}: IOSHeaderProps) => {
  const navigation = useNavigation();
  const router = useRouter();
  const { theme } = useTheme();

  const handleBack = () => {
    if (navigation.canGoBack()) {
      navigation.goBack();
    } else {
      router.back();
    }
  };

  return (
    <Animated.View 
      style={[styles.container, { backgroundColor: theme.background }]}
      entering={FadeIn.duration(300).delay(50)}
    >
      <View style={styles.headerContent}>
        {showBackButton && (
          <TouchableOpacity 
            style={styles.backButton}
            onPress={handleBack}
            activeOpacity={0.7}
          >
            <ArrowLeft size={22} color={theme.primary} />
            <Text style={[styles.backText, { color: theme.primary }]}>Back</Text>
          </TouchableOpacity>
        )}
        
        {!largeTitleStyle && (
          <Text style={[styles.smallTitle, { color: theme.text }]}>{title}</Text>
        )}
        
        {rightComponent && (
          <View style={styles.rightComponent}>
            {rightComponent}
          </View>
        )}
      </View>
      
      {largeTitleStyle && (
        <View style={styles.largeTitleContainer}>
          <Text style={[styles.largeTitle, { color: theme.text }]}>{title}</Text>
        </View>
      )}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingTop: Platform.OS === 'ios' ? 4 : 2,
    zIndex: 10,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 44,
    paddingHorizontal: 16,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 4,
  },
  backText: {
    fontSize: 17,
    marginLeft: 4,
    fontWeight: '500',
  },
  smallTitle: {
    fontSize: 17,
    fontWeight: '600',
    textAlign: 'center',
    position: 'absolute',
    left: 0,
    right: 0,
    alignSelf: 'center',
  },
  rightComponent: {
    marginLeft: 'auto',
  },
  largeTitleContainer: {
    paddingHorizontal: 20,
    paddingTop: 0,
    paddingBottom: 8,
  },
  largeTitle: {
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: -0.5,
  }
}); 