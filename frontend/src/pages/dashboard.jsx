import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import { useLanguage } from '../contexts/LanguageContext';
import { useContent } from '../contexts/ContentContext';
import { useChat } from '../contexts/ChatContext';

const Dashboard = () => {
  const router = useRouter();
  const { translate, currentLanguage } = useLanguage();
  const { contentItems, fetchContentItems } = useContent();
  const { chatSessions, fetchChatSessions } = useChat();
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState({
    totalContent: 0,
    totalChats: 0,
    totalDevices: 0,
    recentActivity: []
  });

  useEffect(() => {
    const loadDashboardData = async () => {
      setIsLoading(true);
      try {
        // Fetch content items and chat sessions in parallel
        await Promise.all([
          fetchContentItems(),
          fetchChatSessions()
        ]);
        
        // Fetch dashboard stats
        const token = localStorage.getItem('token');
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/dashboard/stats`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        }
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadDashboardData();
  }, []);

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString(currentLanguage, { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Get activity icon based on type
  const getActivityIcon = (type) => {
    switch (type) {
      case 'content_upload':
        return (
          <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-md bg-primary-light text-primary">
            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
        );
      case 'chat_session':
        return (
          <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-md bg-primary-light text-primary">
            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
        );
      case 'device_sync':
        return (
          <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-md bg-primary-light text-primary">
            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </div>
        );
      case 'device_register':
        return (
          <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-md bg-primary-light text-primary">
            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
          </div>
        );
      default:
        return (
          <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-md bg-primary-light text-primary">
            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        );
    }
  };

  return (
    <Layout>
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <h1 className="text-2xl font-semibold text-secondary">Dashboard</h1>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          {/* Stats cards */}
          <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {/* Content card */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-primary-light rounded-md p-3">
                    <svg className="h-6 w-6 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Total Content
                      </dt>
                      <dd>
                        <div className="text-lg font-medium text-gray-900">
                          {isLoading ? (
                            <div className="animate-pulse h-6 w-12 bg-gray-200 rounded"></div>
                          ) : (
                            stats.totalContent
                          )}
                        </div>
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-4 sm:px-6">
                <div className="text-sm">
                  <a href="/content" className="font-medium text-primary hover:text-primary-dark">
                    View all content
                    <span aria-hidden="true"> &rarr;</span>
                  </a>
                </div>
              </div>
            </div>

            {/* Chats card */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-primary-light rounded-md p-3">
                    <svg className="h-6 w-6 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Chat Sessions
                      </dt>
                      <dd>
                        <div className="text-lg font-medium text-gray-900">
                          {isLoading ? (
                            <div className="animate-pulse h-6 w-12 bg-gray-200 rounded"></div>
                          ) : (
                            stats.totalChats
                          )}
                        </div>
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-4 sm:px-6">
                <div className="text-sm">
                  <a href="/chat" className="font-medium text-primary hover:text-primary-dark">
                    View all chats
                    <span aria-hidden="true"> &rarr;</span>
                  </a>
                </div>
              </div>
            </div>

            {/* Devices card */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-primary-light rounded-md p-3">
                    <svg className="h-6 w-6 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Connected Devices
                      </dt>
                      <dd>
                        <div className="text-lg font-medium text-gray-900">
                          {isLoading ? (
                            <div className="animate-pulse h-6 w-12 bg-gray-200 rounded"></div>
                          ) : (
                            stats.totalDevices
                          )}
                        </div>
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-4 sm:px-6">
                <div className="text-sm">
                  <a href="/devices" className="font-medium text-primary hover:text-primary-dark">
                    View all devices
                    <span aria-hidden="true"> &rarr;</span>
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Recent activity */}
          <div className="mt-8">
            <h2 className="text-lg leading-6 font-medium text-gray-900">Recent Activity</h2>
            <div className="mt-2 bg-white shadow overflow-hidden sm:rounded-md">
              <ul className="divide-y divide-gray-200">
                {isLoading ? (
                  // Loading skeleton
                  Array(5).fill().map((_, index) => (
                    <li key={index}>
                      <div className="px-4 py-4 sm:px-6">
                        <div className="flex items-center">
                          <div className="animate-pulse flex-shrink-0 h-10 w-10 rounded-md bg-gray-200"></div>
                          <div className="ml-4 flex-1">
                            <div className="animate-pulse h-4 w-3/4 bg-gray-200 rounded mb-2"></div>
                            <div className="animate-pulse h-3 w-1/2 bg-gray-200 rounded"></div>
                          </div>
                          <div className="animate-pulse h-3 w-20 bg-gray-200 rounded"></div>
                        </div>
                      </div>
                    </li>
                  ))
                ) : stats.recentActivity.length > 0 ? (
                  stats.recentActivity.map((activity, index) => (
                    <li key={index}>
                      <div className="px-4 py-4 sm:px-6">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center">
                            {getActivityIcon(activity.type)}
                            <div className="ml-4">
                              <div className="text-sm font-medium text-primary">{activity.title}</div>
                              <div className="text-sm text-gray-500">{activity.description}</div>
                            </div>
                          </div>
                          <div className="text-sm text-gray-500">
                            {formatDate(activity.timestamp)}
                          </div>
                        </div>
                      </div>
                    </li>
                  ))
                ) : (
                  <li className="px-4 py-5 text-center text-gray-500">
                    No recent activity found.
                  </li>
                )}
              </ul>
            </div>
          </div>

          {/* Quick actions */}
          <div className="mt-8">
            <h2 className="text-lg leading-6 font-medium text-gray-900">Quick Actions</h2>
            <div className="mt-2 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
              <div className="bg-white overflow-hidden shadow rounded-lg divide-y divide-gray-200">
                <div className="px-4 py-5 sm:px-6">
                  <h3 className="text-lg font-medium text-gray-900">Upload Content</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Add new documents, audio, or video content for children.
                  </p>
                </div>
                <div className="px-4 py-4 sm:px-6">
                  <a
                    href="/content/upload"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                  >
                    Upload Content
                  </a>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg divide-y divide-gray-200">
                <div className="px-4 py-5 sm:px-6">
                  <h3 className="text-lg font-medium text-gray-900">Start Chat</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Begin a new conversation with EchoNest AI.
                  </p>
                </div>
                <div className="px-4 py-4 sm:px-6">
                  <a
                    href="/chat"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                  >
                    Start Chat
                  </a>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg divide-y divide-gray-200">
                <div className="px-4 py-5 sm:px-6">
                  <h3 className="text-lg font-medium text-gray-900">Register Device</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Add a new device to sync content with.
                  </p>
                </div>
                <div className="px-4 py-4 sm:px-6">
                  <a
                    href="/devices/register"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                  >
                    Register Device
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Dashboard;
