import { useState, useEffect, useCallback } from 'react';
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
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Download, Trash2 } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import * as MediaLibrary from 'expo-media-library';
import { useTheme } from '@/context/ThemeContext';
import { getProcessedImages, downloadProcessedImage, ProcessedImage } from '@/api';
import { triggerHaptic, triggerHapticNotification } from '@/utils/haptics';

export default function HistoryScreen() {
  const [images, setImages] = useState<ProcessedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { theme } = useTheme();

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

  const handleDownload = useCallback(async (imageUrl: string, filename: string) => {
    try {
      triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
      
      // Request permission
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        console.log('Permission not granted');
        return;
      }
      
      // Download the image
      const fileUri = await downloadProcessedImage(imageUrl, filename);
      
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

  if (loading) {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: theme.background }}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.primary} />
          <Text style={[styles.loadingText, { color: theme.text }]}>
            Loading images...
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
        <Text style={[styles.title, { color: theme.text }]}>Processing History</Text>
        
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
          <View style={styles.imagesGrid}>
            {images.map((image, index) => (
              <View 
                key={image.name + index}
                style={[styles.imageCard, { backgroundColor: theme.card }]}
              >
                <Image
                  source={{ uri: image.url }}
                  style={styles.image}
                  resizeMode="cover"
                />
                <View style={styles.imageFooter}>
                  <Text style={[styles.imageName, { color: theme.text }]} numberOfLines={1}>
                    {image.name}
                  </Text>
                  <View style={styles.actionButtons}>
                    <TouchableOpacity
                      style={styles.actionButton}
                      onPress={() => handleDownload(image.url, image.name)}
                    >
                      <Download size={20} color={theme.primary} />
                    </TouchableOpacity>
                  </View>
                </View>
              </View>
            ))}
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
  imagesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  imageCard: {
    width: '48%',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
  },
  image: {
    width: '100%',
    height: 180,
  },
  imageFooter: {
    padding: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  imageName: {
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
  actionButtons: {
    flexDirection: 'row',
  },
  actionButton: {
    padding: 4,
    marginLeft: 8,
  },
});