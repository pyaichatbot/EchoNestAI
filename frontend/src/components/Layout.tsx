import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import LanguageSelector from './LanguageSelector';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [open, setOpen] = React.useState(true);
  const { isAuthenticated, user, logout } = useAuth();
  const { currentLanguage } = useLanguage();
  const router = useRouter();

  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  // Navigation items
  const navItems = [
    { text: 'Dashboard', icon: 'dashboard', path: '/dashboard' },
    { text: 'Chat', icon: 'chat', path: '/chat' },
    { text: 'Content', icon: 'folder', path: '/content' },
    { text: 'Devices', icon: 'devices', path: '/devices' },
    { text: 'Settings', icon: 'settings', path: '/settings' },
  ];

  // Check if we're on a public page (login, register)
  const isPublicPage = ['/login', '/register', '/forgot-password'].includes(router.pathname);

  useEffect(() => {
    if (!isAuthenticated && !isPublicPage) {
      // Redirect to login if not authenticated and not on a public page
      router.push('/login');
    }
  }, [isAuthenticated, isPublicPage, router]);

  if (!isAuthenticated && !isPublicPage) {
    // Return null or a loading indicator while redirecting
    return null;
  }

  if (isPublicPage) {
    // Render a simpler layout for public pages
    return (
      <div className="container mx-auto max-w-md">
        <div className="mt-16 flex flex-col items-center">
          {children}
        </div>
      </div>
    );
  }

  // Icons mapping
  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'dashboard':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        );
      case 'chat':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
        );
      case 'folder':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
          </svg>
        );
      case 'devices':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        );
      case 'settings':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        );
      case 'logout':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 bg-primary-600 text-white shadow-md z-10">
        <div className="flex items-center justify-between px-4 py-2">
          <div className="flex items-center">
            <button
              className="p-1 mr-2 rounded hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-white"
              onClick={handleDrawerToggle}
              aria-label="toggle menu"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-xl font-semibold">EchoNest AI</h1>
          </div>
          <LanguageSelector />
        </div>
      </header>

      {/* Sidebar */}
      <aside className={`fixed top-0 left-0 h-full bg-white shadow-lg transition-all duration-300 ease-in-out z-0 ${open ? 'w-60' : 'w-0'}`} style={{ marginTop: '48px' }}>
        <div className="h-full flex flex-col">
          <nav className="flex-grow overflow-y-auto py-4">
            <ul>
              {navItems.map((item) => (
                <li key={item.text}>
                  <button
                    onClick={() => router.push(item.path)}
                    className={`flex items-center w-full px-4 py-3 text-left ${
                      router.pathname === item.path
                        ? 'bg-primary-50 text-primary-600 border-r-4 border-primary-600'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <span className="mr-3 text-gray-500">{getIcon(item.icon)}</span>
                    <span>{item.text}</span>
                  </button>
                </li>
              ))}
            </ul>
          </nav>
          
          <div className="border-t border-gray-200">
            <button
              onClick={logout}
              className="flex items-center w-full px-4 py-3 text-left text-gray-700 hover:bg-gray-100"
            >
              <span className="mr-3 text-gray-500">{getIcon('logout')}</span>
              <span>Logout</span>
            </button>
            
            {user && (
              <div className="p-4 text-center border-t border-gray-200">
                <p className="text-sm text-gray-500">Logged in as</p>
                <p className="font-medium">
                  {user.firstName} {user.lastName}
                </p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className={`flex-grow transition-all duration-300 ease-in-out ${open ? 'ml-60' : 'ml-0'}`} style={{ marginTop: '48px' }}>
        <div className="p-6">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
