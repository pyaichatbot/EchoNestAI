module.exports = {
  reactStrictMode: true,
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
    API_VERSION: process.env.API_VERSION || 'v1',
  },
  i18n: {
    locales: ['en', 'te', 'ta', 'de', 'hi'],
    defaultLocale: 'en',
    localeDetection: true,
  },
  images: {
    domains: ['localhost'],
  },
  async redirects() {
    return [
      {
        source: '/',
        destination: '/dashboard',
        permanent: true,
      },
    ];
  },
}
