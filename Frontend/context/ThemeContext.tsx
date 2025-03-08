import React, { createContext, useState, useContext, useEffect } from 'react';
import { useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Define theme colors
export const lightTheme = {
  background: '#F9F9F9',
  card: '#FFFFFF',
  text: '#000000',
  textSecondary: '#8E8E93',
  border: '#E5E5EA',
  primary: '#007AFF',
  success: '#34C759',
  placeholder: '#C7C7CC',
  error: '#FF3B30',
};

export const darkTheme = {
  background: '#1C1C1E',
  card: '#2C2C2E',
  text: '#FFFFFF',
  textSecondary: '#8E8E93',
  border: '#38383A',
  primary: '#0A84FF',
  success: '#30D158',
  placeholder: '#636366',
  error: '#FF453A',
};

// Theme context
type ThemeContextType = {
  isDarkMode: boolean;
  setIsDarkMode: (value: boolean) => void;
  toggleTheme: () => void;
  theme: typeof lightTheme;
  isSystemTheme: boolean;
  setIsSystemTheme: (value: boolean) => void;
};

export const ThemeContext = createContext<ThemeContextType>({
  isDarkMode: false,
  setIsDarkMode: () => {},
  toggleTheme: () => {},
  theme: lightTheme,
  isSystemTheme: true,
  setIsSystemTheme: () => {},
});

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const systemColorScheme = useColorScheme();
  const [isSystemTheme, setIsSystemTheme] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(systemColorScheme === 'dark');

  // Load saved theme preferences
  useEffect(() => {
    const loadThemePreferences = async () => {
      try {
        const storedIsSystemTheme = await AsyncStorage.getItem('@isSystemTheme');
        const storedIsDarkMode = await AsyncStorage.getItem('@isDarkMode');
        
        if (storedIsSystemTheme !== null) {
          setIsSystemTheme(storedIsSystemTheme === 'true');
        }
        
        if (storedIsDarkMode !== null && storedIsSystemTheme === 'false') {
          setIsDarkMode(storedIsDarkMode === 'true');
        } else if (storedIsSystemTheme === 'false' && storedIsDarkMode === null) {
          // Default to device preference if there's no stored preference but system theme is off
          setIsDarkMode(systemColorScheme === 'dark');
        }
      } catch (error) {
        console.error('Error loading theme preferences:', error);
      }
    };
    
    loadThemePreferences();
  }, [systemColorScheme]);

  // Update theme based on system changes if system theme is enabled
  useEffect(() => {
    if (isSystemTheme) {
      setIsDarkMode(systemColorScheme === 'dark');
    }
  }, [systemColorScheme, isSystemTheme]);

  // Save theme preferences
  useEffect(() => {
    const saveThemePreferences = async () => {
      try {
        await AsyncStorage.setItem('@isSystemTheme', String(isSystemTheme));
        if (!isSystemTheme) {
          await AsyncStorage.setItem('@isDarkMode', String(isDarkMode));
        }
      } catch (error) {
        console.error('Error saving theme preferences:', error);
      }
    };
    
    saveThemePreferences();
  }, [isSystemTheme, isDarkMode]);

  const toggleTheme = () => {
    if (isSystemTheme) {
      // When switching from system theme to manual, initialize with current state
      setIsSystemTheme(false);
    } else {
      // When in manual mode, just toggle the theme
      setIsDarkMode(prev => !prev);
    }
  };

  const theme = isDarkMode ? darkTheme : lightTheme;

  return (
    <ThemeContext.Provider 
      value={{ 
        isDarkMode, 
        setIsDarkMode, 
        toggleTheme, 
        theme,
        isSystemTheme,
        setIsSystemTheme
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext); 