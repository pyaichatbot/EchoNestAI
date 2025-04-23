import React from 'react';
import { AppProps } from 'next/app';
import { AuthProvider } from '../contexts/AuthContext';
import { LanguageProvider } from '../contexts/LanguageContext';
import { DeviceProvider } from '../contexts/DeviceContext';
import { ContentProvider } from '../contexts/ContentContext';
import { ChatProvider } from '../contexts/ChatContext';
import Layout from '../components/Layout';
import { appWithTranslation } from 'next-i18next';
import '../styles/globals.css';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <AuthProvider>
      <LanguageProvider>
        <DeviceProvider>
          <ContentProvider>
            <ChatProvider>
              <Layout>
                <Component {...pageProps} />
              </Layout>
            </ChatProvider>
          </ContentProvider>
        </DeviceProvider>
      </LanguageProvider>
    </AuthProvider>
  );
}

export default appWithTranslation(MyApp);
