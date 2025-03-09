import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  RefreshControl,
  ActivityIndicator,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { 
  Clock, 
  Info, 
  Image as ImageIcon, 
  Calendar, 
  FileText, 
  Tag,
  AlertTriangle,
  CheckCircle,
  Trash2
} from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import { useTheme } from '@/context/ThemeContext';
import { getProcessedImages, deleteProcessedImage } from '@/api';
import { ProcessedImage } from '@/api/types';
import { triggerHaptic } from '@/utils/haptics';

export default function HistoryScreen() {
  const [images, setImages] = useState<ProcessedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeImage, setActiveImage] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const { theme, isDarkMode } = useTheme();

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

  // Toggle image details
  const toggleImageDetails = useCallback((imageId: string) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setActiveImage(activeImage === imageId ? null : imageId);
  }, [activeImage]);

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
  }, []);

  // Extract model name for display
  const getModelDisplayName = useCallback((image: ProcessedImage) => {
    if (image.modelTypeName) return image.modelTypeName;
    if (image.modelType === 'yolov8m-seg') return 'Segmentation';
    if (image.modelType === 'yolov8m') return 'Detection';
    
    // Extract from filename
    if (image.name.includes('_ms')) return 'Segmentation';
    if (image.name.includes('_m')) return 'Detection';
    
    return 'Detection'; // Default
  }, []);

  if (loading) {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: theme.background }}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.primary} />
          <Text style={[styles.loadingText, { color: theme.text }]}>
            Loading processing history...
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
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
        <Text style={[styles.title, { color: theme.text }]}>History</Text>
        
        {error && (
          <Text style={[styles.errorText, { color: theme.error }]}>
            {error}
          </Text>
        )}
        
        {images.length === 0 && !error ? (
          <View style={styles.emptyContainer}>
            <Text style={[styles.emptyText, { color: theme.textSecondary }]}>
              No processed images yet
            </Text>
            <Text style={[styles.emptySubtext, { color: theme.textSecondary }]}>
              Images will appear here after processing
            </Text>
          </View>
        ) : (
          <View style={styles.historyList}>
            {images.map((image, index) => {
              // Extract model information
              const modelDisplayName = getModelDisplayName(image);
              const isSegmentation = modelDisplayName === 'Segmentation';
              
              // Format processing time
              const processingTimeDisplay = image.processingTime 
                ? `${image.processingTime}s` 
                : 'Quick process';
              
              // Format date
              const processedDate = image.processedDate || 'Recent';
              
              // Check if this image is being deleted
              const isDeleting = deleting === image.name;
              
              return (
                <View 
                  key={image.name + index}
                  style={[
                    styles.historyCard, 
                    { backgroundColor: theme.card }
                  ]}
                >
                  <View style={styles.historyCardHeader}>
                    <View style={styles.historyCardLeft}>
                      <Image
                        source={{ uri: image.url }}
                        style={styles.thumbnail}
                        resizeMode="cover"
                      />
                    </View>
                    <View style={styles.historyCardInfo}>
                      <Text style={[styles.historyCardTitle, { color: theme.text }]} numberOfLines={1}>
                        {image.name.split('_')[0]}
                      </Text>
                      
                      <View style={styles.historyCardTagRow}>
                        <View style={[
                          styles.tag, 
                          { 
                            backgroundColor: isSegmentation
                              ? '#6200ea30' 
                              : '#00796b30'
                          }
                        ]}>
                          <Text style={[
                            styles.tagText, 
                            { 
                              color: isSegmentation
                                ? '#6200ea' 
                                : '#00796b'
                            }
                          ]}>
                            {modelDisplayName}
                          </Text>
                        </View>
                        
                        {image.isFallback && (
                          <View style={[styles.tag, { backgroundColor: '#f4433630' }]}>
                            <Text style={[styles.tagText, { color: '#f44336' }]}>
                              Fallback
                            </Text>
                          </View>
                        )}
                      </View>
                      
                      <View style={styles.historyCardDetailRow}>
                        <Clock size={14} color={theme.textSecondary} />
                        <Text style={[styles.historyCardDetailText, { color: theme.textSecondary }]}>
                          {processingTimeDisplay}
                        </Text>
                      </View>
                      
                      <View style={styles.historyCardDetailRow}>
                        <Calendar size={14} color={theme.textSecondary} />
                        <Text style={[styles.historyCardDetailText, { color: theme.textSecondary }]}>
                          {processedDate}
                        </Text>
                      </View>
                    </View>
                    
                    <View style={styles.historyCardActions}>
                      <TouchableOpacity
                        style={[styles.iconButton, { backgroundColor: isDarkMode ? theme.card : '#f5f5f5' }]}
                        onPress={() => toggleImageDetails(image.name)}
                        disabled={isDeleting}
                      >
                        <Info size={18} color={theme.primary} />
                      </TouchableOpacity>
                      
                      <TouchableOpacity
                        style={[styles.iconButton, { backgroundColor: isDarkMode ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.08)' }]}
                        onPress={() => handleDelete(image)}
                        disabled={isDeleting}
                      >
                        {isDeleting ? (
                          <ActivityIndicator size="small" color="#F44336" />
                        ) : (
                          <Trash2 size={18} color="#F44336" />
                        )}
                      </TouchableOpacity>
                    </View>
                  </View>
                  
                  {activeImage === image.name && (
                    <View style={[styles.detailsContainer, { borderTopColor: theme.border }]}>
                      <Text style={[styles.detailsTitle, { color: theme.text }]}>
                        Image Details
                      </Text>
                      
                      <View style={styles.detailsRow}>
                        <View style={styles.detailItem}>
                          <View style={styles.detailItemHeader}>
                            <FileText size={14} color={theme.primary} />
                            <Text style={[styles.detailItemLabel, { color: theme.text }]}>
                              File Size
                            </Text>
                          </View>
                          <Text style={[styles.detailItemValue, { color: theme.textSecondary }]}>
                            {image.fileSizeFormatted || (image.fileSize ? `${Math.round(image.fileSize / 1024)} KB` : 'Not available')}
                          </Text>
                        </View>
                        
                        <View style={styles.detailItem}>
                          <View style={styles.detailItemHeader}>
                            <ImageIcon size={14} color={theme.primary} />
                            <Text style={[styles.detailItemLabel, { color: theme.text }]}>
                              Dimensions
                            </Text>
                          </View>
                          <Text style={[styles.detailItemValue, { color: theme.textSecondary }]}>
                            {image.width && image.height ? `${image.width} × ${image.height}` : 'Standard size'}
                          </Text>
                        </View>
                      </View>
                      
                      <View style={styles.detailsRow}>
                        <View style={styles.detailItem}>
                          <View style={styles.detailItemHeader}>
                            <Tag size={14} color={theme.primary} />
                            <Text style={[styles.detailItemLabel, { color: theme.text }]}>
                              Model
                            </Text>
                          </View>
                          <Text style={[styles.detailItemValue, { color: theme.textSecondary }]}>
                            {image.modelType || (isSegmentation ? 'YOLOv8m-seg' : 'YOLOv8m')}
                          </Text>
                        </View>
                        
                        <View style={styles.detailItem}>
                          <View style={styles.detailItemHeader}>
                            <Clock size={14} color={theme.primary} />
                            <Text style={[styles.detailItemLabel, { color: theme.text }]}>
                              Processed
                            </Text>
                          </View>
                          <Text style={[styles.detailItemValue, { color: theme.textSecondary }]}>
                            {image.processedTime || 'Today'}
                          </Text>
                        </View>
                      </View>
                      
                      <View style={[styles.statusContainer, { backgroundColor: image.isFallback ? '#fff3e0' : '#e8f5e9' }]}>
                        {image.isFallback ? (
                          <>
                            <AlertTriangle size={16} color="#ff9800" />
                            <Text style={[styles.statusText, { color: '#e65100' }]}>
                              This image was processed using a fallback method
                            </Text>
                          </>
                        ) : (
                          <>
                            <CheckCircle size={16} color="#4caf50" />
                            <Text style={[styles.statusText, { color: '#2e7d32' }]}>
                              Successfully processed with {isSegmentation ? 'segmentation' : 'detection'} model
                            </Text>
                          </>
                        )}
                      </View>
                    </View>
                  )}
                </View>
              );
            })}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: -0.5,
    marginBottom: 20,
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
  historyList: {
    marginBottom: 20,
  },
  historyCard: {
    borderRadius: 16,
    marginBottom: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  historyCardHeader: {
    flexDirection: 'row',
    padding: 12,
  },
  historyCardLeft: {
    marginRight: 12,
  },
  thumbnail: {
    width: 70,
    height: 70,
    borderRadius: 8,
  },
  historyCardInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  historyCardTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  historyCardTagRow: {
    flexDirection: 'row',
    marginBottom: 6,
  },
  tag: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    marginRight: 8,
  },
  tagText: {
    fontSize: 12,
    fontWeight: '500',
  },
  historyCardDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 3,
  },
  historyCardDetailText: {
    fontSize: 13,
    marginLeft: 5,
  },
  historyCardActions: {
    justifyContent: 'center',
    alignItems: 'center',
    paddingLeft: 8,
  },
  iconButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  detailsContainer: {
    padding: 12,
    borderTopWidth: 1,
  },
  detailsTitle: {
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 10,
  },
  detailsRow: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  detailItem: {
    flex: 1,
  },
  detailItemHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  detailItemLabel: {
    fontSize: 13,
    fontWeight: '500',
    marginLeft: 4,
  },
  detailItemValue: {
    fontSize: 13,
    paddingLeft: 18,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderRadius: 8,
    marginTop: 5,
  },
  statusText: {
    fontSize: 13,
    marginLeft: 8,
  },
});