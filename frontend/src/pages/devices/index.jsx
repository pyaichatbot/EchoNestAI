import React, { useState, useEffect } from 'react';
import Layout from '../../components/Layout';
import { useDevice } from '../../contexts/DeviceContext';
import { useLanguage } from '../../contexts/LanguageContext';

const DevicesPage = () => {
  const { devices, fetchDevices, deleteDevice, syncDevice } = useDevice();
  const { translate } = useLanguage();
  const [isLoading, setIsLoading] = useState(true);
  const [syncStatus, setSyncStatus] = useState({});
  const [eventSources, setEventSources] = useState({});

  // Fetch devices on mount
  useEffect(() => {
    const loadDevices = async () => {
      setIsLoading(true);
      await fetchDevices();
      setIsLoading(false);
    };
    
    loadDevices();
  }, []);

  // Clean up event sources on unmount
  useEffect(() => {
    return () => {
      Object.values(eventSources).forEach(es => es.close());
    };
  }, [eventSources]);

  // Handle device deletion
  const handleDelete = async (deviceId) => {
    if (window.confirm('Are you sure you want to delete this device? This will remove all device data and cannot be undone.')) {
      await deleteDevice(deviceId);
    }
  };

  // Handle device sync
  const handleSync = async (deviceId) => {
    try {
      setSyncStatus(prev => ({
        ...prev,
        [deviceId]: { status: 'starting', message: 'Starting sync...' }
      }));
      
      const result = await syncDevice(deviceId);
      
      if (result.success) {
        setSyncStatus(prev => ({
          ...prev,
          [deviceId]: { status: 'syncing', message: 'Sync in progress...' }
        }));
        
        // Connect to SSE for sync updates
        connectToSyncEvents(deviceId);
      } else {
        setSyncStatus(prev => ({
          ...prev,
          [deviceId]: { status: 'error', message: result.message || 'Failed to start sync' }
        }));
      }
    } catch (err) {
      console.error('Error syncing device:', err);
      setSyncStatus(prev => ({
        ...prev,
        [deviceId]: { status: 'error', message: err.message || 'An error occurred' }
      }));
    }
  };

  // Connect to SSE for sync updates
  const connectToSyncEvents = (deviceId) => {
    const token = localStorage.getItem('token');
    const sse = new EventSource(
      `${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices/${deviceId}/sync-events?token=${token}`
    );
    
    sse.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      setSyncStatus(prev => ({
        ...prev,
        [deviceId]: {
          status: data.status,
          message: data.message,
          progress: data.progress || 0,
          details: data.details
        }
      }));
      
      if (data.status === 'completed' || data.status === 'failed') {
        sse.close();
        setEventSources(prev => {
          const newSources = { ...prev };
          delete newSources[deviceId];
          return newSources;
        });
      }
    };
    
    sse.onerror = () => {
      setSyncStatus(prev => ({
        ...prev,
        [deviceId]: {
          ...prev[deviceId],
          status: 'warning',
          message: 'Lost connection to sync updates. The process will continue in the background.'
        }
      }));
      sse.close();
      setEventSources(prev => {
        const newSources = { ...prev };
        delete newSources[deviceId];
        return newSources;
      });
    };
    
    setEventSources(prev => ({
      ...prev,
      [deviceId]: sse
    }));
  };

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  // Get status badge color
  const getStatusColor = (status) => {
    switch (status) {
      case 'online':
        return 'bg-green-100 text-green-800';
      case 'offline':
        return 'bg-red-100 text-red-800';
      case 'syncing':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Get sync status badge color
  const getSyncStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'syncing':
      case 'starting':
        return 'bg-blue-100 text-blue-800';
      case 'error':
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Layout>
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-semibold text-secondary">Devices</h1>
            <a
              href="/devices/register"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            >
              Register New Device
            </a>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-6">
          {/* Devices list */}
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <ul className="divide-y divide-gray-200">
              {isLoading ? (
                <li className="px-4 py-4 sm:px-6 flex justify-center">
                  <svg className="animate-spin h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </li>
              ) : devices.length > 0 ? (
                devices.map((device) => (
                  <li key={device.id}>
                    <div className="px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="flex-shrink-0">
                            <svg className="h-10 w-10 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                            </svg>
                          </div>
                          <div className="ml-4">
                            <div className="flex items-center">
                              <h3 className="text-lg font-medium text-primary truncate">
                                {device.name}
                              </h3>
                              <span className={`ml-2 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(device.status)}`}>
                                {device.status}
                              </span>
                            </div>
                            <p className="text-sm text-gray-500">
                              ID: {device.id}
                            </p>
                          </div>
                        </div>
                        <div className="ml-2 flex-shrink-0 flex">
                          <button
                            onClick={() => handleSync(device.id)}
                            className="mr-2 inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-primary-dark bg-primary-light bg-opacity-20 hover:bg-opacity-30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                            disabled={syncStatus[device.id]?.status === 'syncing' || syncStatus[device.id]?.status === 'starting'}
                          >
                            {syncStatus[device.id]?.status === 'syncing' || syncStatus[device.id]?.status === 'starting' ? (
                              <>
                                <svg className="animate-spin -ml-1 mr-1 h-4 w-4 text-primary-dark" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Syncing...
                              </>
                            ) : (
                              <>
                                <svg className="-ml-1 mr-1 h-4 w-4 text-primary-dark" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                                Sync
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => handleDelete(device.id)}
                            className="inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                          >
                            <svg className="-ml-1 mr-1 h-4 w-4 text-red-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                            Delete
                          </button>
                        </div>
                      </div>
                      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
                        <div className="sm:col-span-1">
                          <dt className="text-sm font-medium text-gray-500">Last Seen</dt>
                          <dd className="mt-1 text-sm text-gray-900">{formatDate(device.last_seen)}</dd>
                        </div>
                        <div className="sm:col-span-1">
                          <dt className="text-sm font-medium text-gray-500">Last Synced</dt>
                          <dd className="mt-1 text-sm text-gray-900">{formatDate(device.last_synced)}</dd>
                        </div>
                        <div className="sm:col-span-1">
                          <dt className="text-sm font-medium text-gray-500">Storage Usage</dt>
                          <dd className="mt-1 text-sm text-gray-900">{device.storage_used} / {device.storage_total}</dd>
                        </div>
                      </div>
                      
                      {/* Sync status */}
                      {syncStatus[device.id] && (
                        <div className="mt-4">
                          <div className={`rounded-md p-3 ${getSyncStatusColor(syncStatus[device.id].status)}`}>
                            <div className="flex">
                              <div className="flex-shrink-0">
                                {syncStatus[device.id].status === 'completed' && (
                                  <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                  </svg>
                                )}
                                {(syncStatus[device.id].status === 'syncing' || syncStatus[device.id].status === 'starting') && (
                                  <svg className="animate-spin h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                  </svg>
                                )}
                                {(syncStatus[device.id].status === 'error' || syncStatus[device.id].status === 'failed') && (
                                  <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                  </svg>
                                )}
                                {syncStatus[device.id].status === 'warning' && (
                                  <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                  </svg>
                                )}
                              </div>
                              <div className="ml-3">
                                <p className="text-sm font-medium">
                                  {syncStatus[device.id].message}
                                </p>
                                
                                {/* Progress bar */}
                                {(syncStatus[device.id].status === 'syncing' || syncStatus[device.id].status === 'starting') && syncStatus[device.id].progress !== undefined && (
                                  <div className="mt-2">
                                    <div className="flex justify-between mb-1">
                                      <span className="text-xs font-medium">Progress</span>
                                      <span className="text-xs font-medium">{syncStatus[device.id].progress}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${syncStatus[device.id].progress}%` }}></div>
                                    </div>
                                  </div>
                                )}
                                
                                {/* Details */}
                                {syncStatus[device.id].details && (
                                  <div className="mt-2 text-xs">
                                    <p>Files synced: {syncStatus[device.id].details.files_synced}</p>
                                    <p>Total size: {syncStatus[device.id].details.total_size}</p>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </li>
                ))
              ) : (
                <li className="px-4 py-5 sm:px-6 text-center text-gray-500">
                  No devices registered. Register a new device to get started!
                </li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default DevicesPage;
