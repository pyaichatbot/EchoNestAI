import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

// Language context for managing multi-language support
export const LanguageContext = React.createContext({
  currentLanguage: 'en',
  changeLanguage: () => {},
  supportedLanguages: {},
  translations: {}
});

// Define supported languages
export const SUPPORTED_LANGUAGES = {
  en: { code: 'en', name: 'English' },
  te: { code: 'te', name: 'Telugu' },
  ta: { code: 'ta', name: 'Tamil' },
  de: { code: 'de', name: 'German' },
  hi: { code: 'hi', name: 'Hindi' },
};

export const LanguageProvider = ({ children }) => {
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [translations, setTranslations] = useState({});
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Load language preference on mount
  useEffect(() => {
    const loadLanguagePreference = () => {
      try {
        // Check localStorage first
        const storedLanguage = localStorage.getItem('language');
        
        if (storedLanguage && SUPPORTED_LANGUAGES[storedLanguage]) {
          setCurrentLanguage(storedLanguage);
        } else {
          // Default to browser language if supported
          const browserLang = navigator.language.split('-')[0];
          if (SUPPORTED_LANGUAGES[browserLang]) {
            setCurrentLanguage(browserLang);
          }
        }
      } catch (err) {
        console.error('Error loading language preference:', err);
      } finally {
        setLoading(false);
      }
    };

    loadLanguagePreference();
  }, []);

  // Load translations when language changes
  useEffect(() => {
    const loadTranslations = async () => {
      try {
        // In a production app, we would load translations from a file or API
        // For now, we'll use a simple object with some common translations
        const translationsData = {
          en: {
            common: {
              login: 'Login',
              register: 'Register',
              logout: 'Logout',
              email: 'Email',
              password: 'Password',
              submit: 'Submit',
              cancel: 'Cancel',
              save: 'Save',
              delete: 'Delete',
              loading: 'Loading...',
              error: 'An error occurred',
              success: 'Success',
            },
            nav: {
              dashboard: 'Dashboard',
              chat: 'Chat',
              content: 'Content',
              devices: 'Devices',
              settings: 'Settings',
            },
            chat: {
              newChat: 'New Chat',
              sendMessage: 'Send Message',
              typeMessage: 'Type your message...',
              voiceMessage: 'Voice Message',
              sources: 'Sources',
            },
            content: {
              upload: 'Upload Content',
              title: 'Title',
              description: 'Description',
              language: 'Language',
              tags: 'Tags',
              uploadProgress: 'Upload Progress',
              processing: 'Processing content...',
              completed: 'Upload completed!',
            },
            devices: {
              registerDevice: 'Register Device',
              deviceName: 'Device Name',
              lastSync: 'Last Sync',
              status: 'Status',
              sync: 'Sync',
              syncRequired: 'Sync Required',
              pendingDocuments: 'Pending Documents',
            },
          },
          // Add basic translations for other languages
          te: {
            common: {
              login: 'లాగిన్',
              register: 'నమోదు',
              logout: 'లాగౌట్',
              // Add more translations as needed
            },
            // Add more sections as needed
          },
          ta: {
            common: {
              login: 'உள்நுழைய',
              register: 'பதிவு',
              logout: 'வெளியேறு',
              // Add more translations as needed
            },
            // Add more sections as needed
          },
          de: {
            common: {
              login: 'Anmelden',
              register: 'Registrieren',
              logout: 'Abmelden',
              // Add more translations as needed
            },
            // Add more sections as needed
          },
          hi: {
            common: {
              login: 'लॉगिन',
              register: 'रजिस्टर',
              logout: 'लॉगआउट',
              // Add more translations as needed
            },
            // Add more sections as needed
          },
        };

        setTranslations(translationsData[currentLanguage] || translationsData.en);
      } catch (err) {
        console.error('Error loading translations:', err);
        // Fallback to English translations
        setTranslations({});
      }
    };

    loadTranslations();
  }, [currentLanguage]);

  // Change language function
  const changeLanguage = (lang) => {
    if (SUPPORTED_LANGUAGES[lang]) {
      setCurrentLanguage(lang);
      
      // Store language preference
      try {
        localStorage.setItem('language', lang);
      } catch (err) {
        console.error('Error storing language preference:', err);
      }
      
      // Update document language attribute
      document.documentElement.lang = lang;
      
      // Update URL to reflect language change if using i18n routing
      // This would depend on your routing setup
    }
  };

  // Translate function (to be used in components)
  const translate = (key) => {
    const keys = key.split('.');
    let result = translations;
    
    for (const k of keys) {
      if (result && result[k]) {
        result = result[k];
      } else {
        // Fallback to key if translation not found
        return key;
      }
    }
    
    return result;
  };

  return (
    <LanguageContext.Provider
      value={{
        currentLanguage,
        changeLanguage,
        supportedLanguages: SUPPORTED_LANGUAGES,
        translations,
        translate,
        loading
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
};

// Custom hook to use language context
export const useLanguage = () => {
  const context = React.useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
