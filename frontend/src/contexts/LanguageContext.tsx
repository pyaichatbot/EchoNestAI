import { createContext, useContext, useState, ReactNode } from 'react';
import { useRouter } from 'next/router';
import i18n from 'i18next';

// Define supported languages
export const SUPPORTED_LANGUAGES = {
  en: { code: 'en', name: 'English' },
  te: { code: 'te', name: 'Telugu' },
  ta: { code: 'ta', name: 'Tamil' },
  de: { code: 'de', name: 'German' },
  hi: { code: 'hi', name: 'Hindi' },
};

// Define types
interface LanguageContextType {
  currentLanguage: string;
  changeLanguage: (lang: string) => void;
  supportedLanguages: typeof SUPPORTED_LANGUAGES;
}

interface LanguageProviderProps {
  children: ReactNode;
}

// Create context
const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

// Create provider
export const LanguageProvider = ({ children }: LanguageProviderProps) => {
  const [currentLanguage, setCurrentLanguage] = useState<string>(i18n.language || 'en');
  const router = useRouter();

  // Change language function
  const changeLanguage = (lang: string) => {
    if (SUPPORTED_LANGUAGES[lang as keyof typeof SUPPORTED_LANGUAGES]) {
      i18n.changeLanguage(lang);
      setCurrentLanguage(lang);
      
      // Update the URL to reflect the language change
      const { pathname, asPath, query } = router;
      router.push({ pathname, query }, asPath, { locale: lang });
      
      // Store language preference
      localStorage.setItem('language', lang);
      
      // Update document direction for RTL languages if needed
      document.documentElement.lang = lang;
    }
  };

  return (
    <LanguageContext.Provider
      value={{
        currentLanguage,
        changeLanguage,
        supportedLanguages: SUPPORTED_LANGUAGES,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
};

// Custom hook to use language context
export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
