import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  Dimensions,
  RefreshControl,
  ActivityIndicator,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Download, Eye, Trash2, Layers } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import * as MediaLibrary from 'expo-media-library';
import { useTheme } from '@/context/ThemeContext';
import { getProcessedImages, downloadProcessedImage, deleteProcessedImage } from '@/api';
import { ProcessedImage } from '@/api/types';
import { triggerHaptic, triggerHapticNotification } from '@/utils/haptics';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { PinchGestureView } from '../../components/PinchGestureView';
import ImageAlbumView from '../../components/ImageAlbumView';

const windowWidth = Dimensions.get('window').width;
const imageSize = (windowWidth - 60) / 2;

// Interface for grouped stack images
interface ImageStack {
  id: string;
  images: ProcessedImage[];
  previewImage: ProcessedImage;
  count: number;
}

export default function GalleryScreen() {
  const [images, setImages] = useState<ProcessedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<ProcessedImage | null>(null);
  const [selectedStack, setSelectedStack] = useState<ImageStack | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const { theme, isDarkMode } = useTheme();

  // Group images into stacks based on stackId
  const { stackedImages, singleImages } = useMemo(() => {
    const stacks: Record<string, ProcessedImage[]> = {};
    const singles: ProcessedImage[] = [];

    // First pass: group images by stackId
    images.forEach(image => {
      if (image.stackId && image.isStackImage) {
        if (!stacks[image.stackId]) {
          stacks[image.stackId] = [];
        }
        stacks[image.stackId].push(image);
      } else {
        singles.push(image);
      }
    });

    // Convert to array of ImageStack objects
    const stacksArray: ImageStack[] = Object.entries(stacks).map(([id, stackImages]) => ({
      id,
      images: stackImages,
      previewImage: stackImages[0], // Use first image as preview
      count: stackImages.length
    }));

    return {
      stackedImages: stacksArray,
      singleImages: singles
    };
  }, [images]);

  const fetchImages = useCallback(async () => {
    try {
      setError(null);
      const fetchedImages = await getProcessedImages();
      setImages(fetchedImages);
    } catch (err) {
      console.error('Failed to fetch images:', err);
      setError('Failed to load images. Please try again.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchImages();
  }, [fetchImages]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchImages();
  }, [fetchImages]);

  // Handle image download
  const handleDownload = useCallback(async (image: ProcessedImage) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'This app needs permission to save images to your gallery.',
          [
            { text: 'OK' }
          ]
        );
        return;
      }
      
      // Show loading indicator or toast here
      Alert.alert('Downloading', 'Saving image to your gallery...');
      
      // Use the API to download the image
      const localUri = await downloadProcessedImage(image.url, image.name);
      
      // Save the downloaded image to the media library
      await MediaLibrary.saveToLibraryAsync(localUri);
      
      // Success notification
      Alert.alert('Success', 'Image saved to your gallery!');
      triggerHapticNotification(Haptics.NotificationFeedbackType.Success);
      
    } catch (error) {
      console.error('Download error:', error);
      Alert.alert('Error', 'Failed to download image. Please try again.');
      triggerHapticNotification(Haptics.NotificationFeedbackType.Error);
    }
  }, []);

  // Handle image deletion
  const handleDelete = useCallback((image: ProcessedImage) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    
    Alert.alert(
      'Delete Image',
      `Are you sure you want to delete this image? This action cannot be undone.`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              setDeleting(image.name);
              await deleteProcessedImage(image.name);
              // Remove the image from the state
              setImages(prevImages => prevImages.filter(img => img.name !== image.name));
              // Close preview if this was the selected image
              if (selectedImage && selectedImage.name === image.name) {
                setSelectedImage(null);
              }
              // Close stack preview if this image was in the selected stack
              if (selectedStack && selectedStack.images.some(img => img.name === image.name)) {
                setSelectedStack(null);
              }
              triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
            } catch (error) {
              console.error('Failed to delete image:', error);
              Alert.alert('Error', 'Failed to delete image. Please try again.');
            } finally {
              setDeleting(null);
            }
          },
        },
      ],
      { cancelable: true }
    );
  }, [selectedImage, selectedStack]);

  // Handle deleting an entire stack
  const handleDeleteStack = useCallback((stack: ImageStack) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    
    Alert.alert(
      'Delete Image Stack',
      `Are you sure you want to delete all ${stack.images.length} images in this stack? This action cannot be undone.`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete All',
          style: 'destructive',
          onPress: async () => {
            try {
              // Delete each image in the stack
              const stackImageNames = stack.images.map(img => img.name);
              setDeleting(stack.id); // Use stack ID for deleting state
              
              // Delete images one by one
              for (const imageName of stackImageNames) {
                await deleteProcessedImage(imageName);
              }
              
              // Remove all stack images from state
              setImages(prevImages => prevImages.filter(img => !stackImageNames.includes(img.name)));
              
              // Close stack preview if this was the selected stack
              if (selectedStack && selectedStack.id === stack.id) {
                setSelectedStack(null);
              }
              
              triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
            } catch (error) {
              console.error('Failed to delete stack:', error);
              Alert.alert('Error', 'Failed to delete some images in the stack. Please try again.');
            } finally {
              setDeleting(null);
            }
          },
        },
      ],
      { cancelable: true }
    );
  }, [selectedStack]);

  const handleImagePress = useCallback((image: ProcessedImage) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedImage(image);
    setSelectedStack(null); // Close stack view if open
  }, []);

  const handleStackPress = useCallback((stack: ImageStack) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedStack(stack);
    setSelectedImage(null); // Close single image view if open
  }, []);

  const handleClosePreview = useCallback(() => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedImage(null);
  }, []);

  const handleCloseStackPreview = useCallback(() => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedStack(null);
  }, []);

  if (loading) {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: theme.background }}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.primary} />
          <Text style={[styles.loadingText, { color: theme.text }]}>
            Loading gallery...
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaView style={{ flex: 1, backgroundColor: theme.background }}>
        <ScrollView
          contentContainerStyle={styles.container}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={theme.text}
            />
          }
        >
          <Text style={[styles.title, { color: theme.text }]}>Gallery</Text>
          
          {error && (
            <Text style={[styles.errorText, { color: theme.error }]}>
              {error}
            </Text>
          )}
          
          {images.length === 0 && !error ? (
            <View style={styles.emptyContainer}>
              <Text style={[styles.emptyText, { color: theme.textSecondary }]}>
                Your gallery is empty
              </Text>
              <Text style={[styles.emptySubtext, { color: theme.textSecondary }]}>
                Processed images will appear here
              </Text>
            </View>
          ) : (
            <>
              {/* Stacks section */}
              {stackedImages.length > 0 && (
                <View style={styles.sectionContainer}>
                  <Text style={[styles.sectionTitle, { color: theme.text }]}>
                    Image Stacks
                  </Text>
                  <View style={styles.galleryGrid}>
                    {stackedImages.map((stack) => {
                      const isDeleting = deleting === stack.id;
                      
                      return (
                        <TouchableOpacity
                          key={stack.id}
                          style={[
                            styles.imageCard, 
                            { 
                              backgroundColor: theme.card,
                              shadowColor: isDarkMode ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.2)',
                            }
                          ]}
                          onPress={() => handleStackPress(stack)}
                          activeOpacity={0.8}
                          disabled={isDeleting}
                        >
                          <View style={styles.imageCardInner}>
                            <Image
                              source={{ uri: stack.previewImage.url }}
                              style={[
                                styles.thumbnail,
                                isDeleting && { opacity: 0.5 }
                              ]}
                              resizeMode="cover"
                            />
                            
                            {/* Stack count badge */}
                            <View style={styles.stackBadge}>
                              <Layers size={14} color="#fff" />
                              <Text style={styles.stackCount}>{stack.count}</Text>
                            </View>
                            
                            {isDeleting && (
                              <View style={styles.deletingOverlay}>
                                <ActivityIndicator size="large" color="#fff" />
                              </View>
                            )}
                            
                            <View style={styles.imageOverlay}>
                              <View style={styles.actionButtonsRow}>
                                <TouchableOpacity
                                  style={[styles.actionButton, { 
                                    backgroundColor: isDarkMode ? theme.card : '#f5f5f5'
                                  }]}
                                  onPress={() => handleDeleteStack(stack)}
                                  disabled={isDeleting}
                                >
                                  <Trash2 size={16} color="#F44336" />
                                </TouchableOpacity>
                              </View>
                            </View>
                          </View>
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </View>
              )}
              
              {/* Individual images section */}
              {singleImages.length > 0 && (
                <View style={styles.sectionContainer}>
                  <Text style={[styles.sectionTitle, { color: theme.text }]}>
                    Images
                  </Text>
                  <View style={styles.galleryGrid}>
                    {singleImages.map((image, index) => {
                      const isDeleting = deleting === image.name;
                      
                      return (
                        <TouchableOpacity
                          key={image.name + index}
                          style={[
                            styles.imageCard, 
                            { 
                              backgroundColor: theme.card,
                              shadowColor: isDarkMode ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.2)',
                            }
                          ]}
                          onPress={() => handleImagePress(image)}
                          activeOpacity={0.8}
                          disabled={isDeleting}
                        >
                          <View style={styles.imageCardInner}>
                            <Image
                              source={{ uri: image.url }}
                              style={[
                                styles.thumbnail,
                                isDeleting && { opacity: 0.5 }
                              ]}
                              resizeMode="cover"
                            />
                            
                            {isDeleting && (
                              <View style={styles.deletingOverlay}>
                                <ActivityIndicator size="large" color="#fff" />
                              </View>
                            )}
                            
                            <View style={styles.imageOverlay}>
                              <View style={styles.actionButtonsRow}>
                                <TouchableOpacity
                                  style={[styles.actionButton, { 
                                    backgroundColor: isDarkMode ? theme.card : '#f5f5f5',
                                    marginLeft: 8 
                                  }]}
                                  onPress={() => handleDownload(image)}
                                  disabled={isDeleting}
                                >
                                  <Download size={16} color={theme.primary} />
                                </TouchableOpacity>
                                
                                <TouchableOpacity
                                  style={[styles.actionButton, { 
                                    backgroundColor: isDarkMode ? theme.card : '#f5f5f5',
                                    marginLeft: 8 
                                  }]}
                                  onPress={() => handleDelete(image)}
                                  disabled={isDeleting}
                                >
                                  <Trash2 size={16} color="#F44336" />
                                </TouchableOpacity>
                              </View>
                            </View>
                          </View>
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </View>
              )}
            </>
          )}
        </ScrollView>
        
        {/* Single Image Preview Modal */}
        {selectedImage && (
          <View style={[styles.previewContainer, { backgroundColor: isDarkMode ? 'rgba(0,0,0,0.95)' : 'rgba(0,0,0,0.85)' }]}>
            {/* Top action buttons row */}
            <View style={styles.previewActions}>
              <TouchableOpacity
                style={[styles.actionButton, { 
                  backgroundColor: isDarkMode ? theme.card : '#f5f5f5'
                }]}
                onPress={() => handleClosePreview()}
              >
                <Text style={[styles.closeText, { color: isDarkMode ? '#fff' : '#333' }]}>×</Text>
              </TouchableOpacity>
              
              <View style={styles.rightButtonsGroup}>
                <TouchableOpacity
                  style={[styles.actionButton, { 
                    backgroundColor: isDarkMode ? theme.card : '#f5f5f5',
                    marginLeft: 10,
                  }]}
                  onPress={() => handleDownload(selectedImage)}
                >
                  <Download size={18} color={theme.primary} />
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={[styles.actionButton, { 
                    backgroundColor: isDarkMode ? theme.card : '#f5f5f5',
                    marginLeft: 10,
                  }]}
                  onPress={() => handleDelete(selectedImage)}
                >
                  <Trash2 size={18} color="#F44336" />
                </TouchableOpacity>
              </View>
            </View>
            
            <SafeAreaView style={styles.previewContent}>
              <View style={styles.previewImageContainer}>
                <PinchGestureView>
                  <Image
                    source={{ uri: selectedImage.url }}
                    style={styles.previewImage}
                    resizeMode="contain"
                  />
                </PinchGestureView>
              </View>
              
              <Text style={styles.previewName} numberOfLines={1}>
                {selectedImage.name.split('_')[0]}
              </Text>
              
              <Text style={styles.zoomHint}>
                Pinch to zoom • Double tap to reset
              </Text>
            </SafeAreaView>
          </View>
        )}
        
        {/* Stack Image Album View */}
        {selectedStack && (
          <ImageAlbumView
            images={selectedStack.images}
            onClose={handleCloseStackPreview}
            onDelete={handleDelete}
            onDownload={handleDownload}
            isDarkMode={isDarkMode}
            theme={theme}
          />
        )}
      </SafeAreaView>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
  },
  title: {
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: -0.5,
    marginBottom: 20,
  },
  errorText: {
    marginBottom: 15,
    textAlign: 'center',
    fontSize: 16,
  },
  emptyContainer: {
    padding: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    textAlign: 'center',
  },
  sectionContainer: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 12,
  },
  galleryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  imageCard: {
    width: imageSize,
    height: imageSize,
    borderRadius: 16,
    marginBottom: 20,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  imageCardInner: {
    width: '100%',
    height: '100%',
    borderRadius: 16,
    overflow: 'hidden',
  },
  thumbnail: {
    width: '100%',
    height: '100%',
    borderRadius: 16,
  },
  stackBadge: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 4,
    flexDirection: 'row',
    alignItems: 'center',
  },
  stackCount: {
    color: '#fff',
    marginLeft: 4,
    fontWeight: '600',
    fontSize: 12,
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 10,
    right: 10,
  },
  actionButtonsRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  actionButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  deletingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewContent: {
    flex: 1,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: Platform.OS === 'ios' ? 0 : 20,
  },
  previewActions: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 50 : 30,
    left: 0,
    right: 0,
    zIndex: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    width: '100%',
    height: 36,
  },
  rightButtonsGroup: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  closeText: {
    fontSize: 22,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 24,
  },
  previewImageContainer: {
    width: '100%',
    height: '70%',
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  previewImage: {
    width: '100%',
    height: '100%',
    borderRadius: 16,
  },
  previewName: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 20,
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  zoomHint: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  downloadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 20,
  },
  downloadText: {
    color: 'white',
    fontWeight: 'bold',
    marginLeft: 8,
  },
});