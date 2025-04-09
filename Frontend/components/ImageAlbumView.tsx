import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Image,
  TouchableOpacity,
  FlatList,
  Animated,
  Platform,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';
import { ProcessedImage } from '@/api/types';
import { X, ChevronLeft, ChevronRight, Download, Trash2 } from 'lucide-react-native';
import { PinchGestureView } from './PinchGestureView';
import * as Haptics from 'expo-haptics';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

const { width, height } = Dimensions.get('window');

interface ImageAlbumViewProps {
  images: ProcessedImage[];
  initialIndex?: number;
  onClose: () => void;
  onDownload?: (image: ProcessedImage) => void;
  onDelete?: (image: ProcessedImage) => void;
  isDarkMode: boolean;
  theme: any;
}

const ImageAlbumView: React.FC<ImageAlbumViewProps> = ({
  images,
  initialIndex = 0,
  onClose,
  onDownload,
  onDelete,
  isDarkMode,
  theme,
}) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const flatListRef = useRef<FlatList>(null);
  
  // Handle pagination and ensure current index is valid
  const currentImage = images[currentIndex] || images[0];
  const totalImages = images.length;
  
  // Function to navigate to next image
  const goToNextImage = () => {
    if (currentIndex < totalImages - 1) {
      const nextIndex = currentIndex + 1;
      setCurrentIndex(nextIndex);
      flatListRef.current?.scrollToIndex({
        index: nextIndex,
        animated: true,
      });
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };
  
  // Function to navigate to previous image
  const goToPrevImage = () => {
    if (currentIndex > 0) {
      const prevIndex = currentIndex - 1;
      setCurrentIndex(prevIndex);
      flatListRef.current?.scrollToIndex({
        index: prevIndex,
        animated: true,
      });
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };
  
  // Handler for when scroll ends
  const handleScroll = (event: any) => {
    const viewSize = event.nativeEvent.layoutMeasurement.width;
    const contentOffset = event.nativeEvent.contentOffset.x;
    
    const newIndex = Math.floor(contentOffset / viewSize);
    if (newIndex !== currentIndex) {
      setCurrentIndex(newIndex);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };
  
  // Render individual image item
  const renderItem = ({ item }: { item: ProcessedImage }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);
    
    return (
      <View style={styles.imageItemContainer}>
        <PinchGestureView>
          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#6A35D9" />
              <Text style={styles.loadingText}>Loading image...</Text>
            </View>
          )}
          
          {hasError && (
            <View style={styles.errorContainer}>
              <X size={40} color="#F44336" />
              <Text style={styles.errorText}>Failed to load image</Text>
              <Text style={styles.errorUrl}>{item.url}</Text>
            </View>
          )}
          
          <Image
            source={{ uri: item.url }}
            style={[
              styles.previewImage,
              (isLoading || hasError) && { opacity: 0 }
            ]}
            resizeMode="contain"
            defaultSource={require('../assets/images/icon.png')}
            onLoadStart={() => {
              setIsLoading(true);
              setHasError(false);
            }}
            onLoad={() => {
              console.log('Image loaded successfully:', item.url);
              setIsLoading(false);
            }}
            onError={(e) => {
              console.error('Error loading image:', e.nativeEvent.error, item.url);
              setIsLoading(false);
              setHasError(true);
            }}
          />
        </PinchGestureView>
      </View>
    );
  };
  
  // Extract image name from the first part before underscore
  const getImageName = (name: string) => {
    return name.split('_')[0] || name;
  };
  
  return (
    <GestureHandlerRootView style={styles.container}>
      <View style={[
        styles.container,
        { backgroundColor: isDarkMode ? 'rgba(0,0,0,0.95)' : 'rgba(0,0,0,0.85)' }
      ]}>
        {/* Top action buttons */}
        <View style={styles.actionBar}>
          <TouchableOpacity
            style={[styles.closeButton, { backgroundColor: isDarkMode ? theme.card : '#f5f5f5' }]}
            onPress={onClose}
          >
            <X size={18} color={isDarkMode ? '#fff' : '#333'} />
          </TouchableOpacity>
          
          <View style={styles.paginationContainer}>
            <Text style={styles.paginationText}>
              {currentIndex + 1} / {totalImages}
            </Text>
          </View>
          
          <View style={styles.rightActions}>
            {onDownload && (
              <TouchableOpacity
                style={[styles.actionButton, { backgroundColor: isDarkMode ? theme.card : '#f5f5f5' }]}
                onPress={() => onDownload(currentImage)}
              >
                <Download size={18} color={theme.primary} />
              </TouchableOpacity>
            )}
            
            {onDelete && (
              <TouchableOpacity
                style={[styles.actionButton, { backgroundColor: isDarkMode ? theme.card : '#f5f5f5' }]}
                onPress={() => onDelete(currentImage)}
              >
                <Trash2 size={18} color="#F44336" />
              </TouchableOpacity>
            )}
          </View>
        </View>
        
        {/* Image gallery */}
        <FlatList
          ref={flatListRef}
          data={images}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          initialScrollIndex={initialIndex}
          getItemLayout={(_, index) => ({
            length: width,
            offset: width * index,
            index,
          })}
          renderItem={renderItem}
          keyExtractor={(item) => item.name}
          onMomentumScrollEnd={handleScroll}
        />
        
        {/* Navigation arrows */}
        {currentIndex > 0 && (
          <TouchableOpacity
            style={[styles.navButton, styles.navButtonLeft]}
            onPress={goToPrevImage}
          >
            <ChevronLeft size={30} color="#fff" />
          </TouchableOpacity>
        )}
        
        {currentIndex < totalImages - 1 && (
          <TouchableOpacity
            style={[styles.navButton, styles.navButtonRight]}
            onPress={goToNextImage}
          >
            <ChevronRight size={30} color="#fff" />
          </TouchableOpacity>
        )}
        
        {/* Bottom info */}
        <View style={styles.bottomInfo}>
          <Text style={styles.imageName} numberOfLines={1}>
            {getImageName(currentImage.name)}
          </Text>
          
          <Text style={styles.zoomHint}>
            Pinch to zoom • Double tap to reset • Swipe to change image
          </Text>
        </View>
      </View>
    </GestureHandlerRootView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  imageItemContainer: {
    width,
    height: height * 0.7,
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  actionBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 15,
    width: '100%',
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 10,
    zIndex: 10,
  },
  closeButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  paginationContainer: {
    backgroundColor: 'rgba(0,0,0,0.3)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
  },
  paginationText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 14,
  },
  rightActions: {
    flexDirection: 'row',
  },
  actionButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  navButton: {
    position: 'absolute',
    top: '50%',
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    transform: [{ translateY: -20 }],
  },
  navButtonLeft: {
    left: 15,
  },
  navButtonRight: {
    right: 15,
  },
  bottomInfo: {
    width: '100%',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
    position: 'absolute',
    bottom: 0,
  },
  imageName: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 10,
    textAlign: 'center',
  },
  zoomHint: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 10,
  },
  errorContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 10,
  },
  errorUrl: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
});

export default ImageAlbumView; 