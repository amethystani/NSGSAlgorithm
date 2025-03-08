import { useState, useEffect, useCallback } from 'react';
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
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Download, Eye } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import * as MediaLibrary from 'expo-media-library';
import { useTheme } from '@/context/ThemeContext';
import { getProcessedImages, downloadProcessedImage, ProcessedImage } from '@/api';
import { triggerHaptic, triggerHapticNotification } from '@/utils/haptics';

const windowWidth = Dimensions.get('window').width;
const imageSize = (windowWidth - 60) / 2; // 2 images per row with padding

export default function GalleryScreen() {
  const [images, setImages] = useState<ProcessedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<ProcessedImage | null>(null);
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

  const handleDownload = useCallback(async (image: ProcessedImage) => {
    try {
      triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
      
      // Request permission
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        console.log('Permission not granted');
        return;
      }
      
      // Download the image
      const fileUri = await downloadProcessedImage(image.url, image.name);
      
      // Save to media library
      await MediaLibrary.saveToLibraryAsync(fileUri);
      
      // Success feedback
      triggerHapticNotification(Haptics.NotificationFeedbackType.Success);
      
      console.log('Image saved to gallery');
    } catch (err) {
      console.error('Error saving image:', err);
      triggerHapticNotification(Haptics.NotificationFeedbackType.Error);
    }
  }, []);

  const handleImagePress = useCallback((image: ProcessedImage) => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedImage(image);
  }, []);

  const handleClosePreview = useCallback(() => {
    triggerHaptic(Haptics.ImpactFeedbackStyle.Light);
    setSelectedImage(null);
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
        <Text style={[styles.title, { color: theme.text }]}>Object Detection Gallery</Text>
        
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
          <View style={styles.galleryGrid}>
            {images.map((image, index) => (
              <TouchableOpacity
                key={image.name + index}
                style={[styles.imageCard, { backgroundColor: theme.card }]}
                onPress={() => handleImagePress(image)}
                activeOpacity={0.8}
              >
                <Image
                  source={{ uri: image.url }}
                  style={styles.thumbnail}
                  resizeMode="cover"
                />
                <View style={styles.imageOverlay}>
                  <TouchableOpacity
                    style={[styles.actionButton, { backgroundColor: theme.primary }]}
                    onPress={() => handleDownload(image)}
                  >
                    <Download size={16} color="#fff" />
                  </TouchableOpacity>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </ScrollView>
      
      {/* Image Preview Modal */}
      {selectedImage && (
        <View style={[styles.previewContainer, { backgroundColor: isDarkMode ? 'rgba(0,0,0,0.9)' : 'rgba(0,0,0,0.7)' }]}>
          <TouchableOpacity
            style={styles.closePreview}
            onPress={handleClosePreview}
          >
            <Text style={styles.closeText}>×</Text>
          </TouchableOpacity>
          
          <View style={styles.previewImageContainer}>
            <Image
              source={{ uri: selectedImage.url }}
              style={styles.previewImage}
              resizeMode="contain"
            />
          </View>
          
          <Text style={styles.previewName}>{selectedImage.name}</Text>
          
          <TouchableOpacity
            style={[styles.downloadButton, { backgroundColor: theme.primary }]}
            onPress={() => handleDownload(selectedImage)}
          >
            <Download size={20} color="#fff" />
            <Text style={styles.downloadText}>Save to Gallery</Text>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
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
    fontSize: 24,
    fontWeight: 'bold',
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
  galleryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  imageCard: {
    width: imageSize,
    height: imageSize,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
  },
  thumbnail: {
    width: '100%',
    height: '100%',
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 10,
    right: 10,
  },
  actionButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
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
    padding: 20,
  },
  closePreview: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
  },
  closeText: {
    color: 'white',
    fontSize: 36,
    fontWeight: 'bold',
  },
  previewImageContainer: {
    width: '100%',
    height: '70%',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  previewName: {
    color: 'white',
    fontSize: 16,
    marginTop: 20,
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