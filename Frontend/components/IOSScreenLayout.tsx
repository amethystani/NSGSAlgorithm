import React, { ReactNode } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { IOSHeader } from './IOSHeader';
import Animated, { FadeIn } from 'react-native-reanimated';
import { useTheme } from '@/context/ThemeContext';

interface IOSScreenLayoutProps {
  children: ReactNode;
  title: string;
  showBackButton?: boolean;
  rightComponent?: ReactNode;
  scrollable?: boolean;
  largeTitleStyle?: boolean;
}

export const IOSScreenLayout = ({
  children,
  title,
  showBackButton = false,
  rightComponent,
  scrollable = true,
  largeTitleStyle = true,
}: IOSScreenLayoutProps) => {
  const { theme } = useTheme();
  
  const Content = () => (
    <>
      <IOSHeader 
        title={title} 
        showBackButton={showBackButton} 
        rightComponent={rightComponent}
        largeTitleStyle={largeTitleStyle}
      />
      <Animated.View 
        style={[styles.contentContainer, { backgroundColor: theme.background }]}
        entering={FadeIn.duration(300).delay(100)}
      >
        {children}
      </Animated.View>
    </>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      {scrollable ? (
        <ScrollView 
          style={styles.scrollView} 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <Content />
        </ScrollView>
      ) : (
        <Content />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: 30,
  },
  contentContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
}); 