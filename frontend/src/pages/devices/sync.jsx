import React, { useState, useEffect } from 'react';
import Layout from '../../components/Layout';
import { useDevice } from '../../contexts/DeviceContext';
import { useLanguage } from '../../contexts/LanguageContext';

const DeviceSyncPage = () => {
  const { devices, fetchDevices, syncAllDevices } = useDevice();
  const { translate } = useLanguage();
  const [isLoading, setIsLoading] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncStatus, setSyncStatus] = useState({
    status: null,
    message: null,
    progress: 0,
    deviceStatuses: {}
  });
  const [eventSource, setEventSource] = useState(null);

  // Fetch devices on mount
  useEffect(() => {
    const loadDevices = async () => {
      setIsLoading(true);
      await fetchDevices();
      setIsLoading(false);
    };
    
    loadDevices();
  }, []);

  // Clean up event source on unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [eventSource]);

  // Handle sync all devices
  const handleSyncAll = async () => {
    try {
      setIsSyncing(true);
      setSyncStatus({
        status: 'starting',
        message: 'Starting sync for all devices...',
        progress: 0,
        deviceStatuses: {}
      });
      
      const result = await syncAllDevices();
      
      if (result.success) {
        setSyncStatus(prev => ({
          ...prev,
          status: 'syncing',
          message: 'Sync in progress...'
        }));
        
        // Connect to SSE for sync updates
        connectToSyncEvents(result.batchId);
      } else {
        setSyncStatus(prev => ({
          ...prev,
          status: 'error',
          message: result.message || 'Failed to start sync'
        }));
        setIsSyncing(false);
      }
    } catch (err) {
      console.error('Error syncing all devices:', err);
      setSyncStatus(prev => ({
        ...prev,
        status: 'error',
        message: err.message || 'An error occurred'
      }));
      setIsSyncing(false);
    }
  };

  // Connect to SSE for sync updates
  const connectToSyncEvents = (batchId) => {
    const token = localStorage.getItem('token');
    const sse = new EventSource(
      `${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices/sync-batch/${batchId}/events?token=${token}`
    );
    
    sse.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      setSyncStatus(prev => ({
        status: data.status,
        message: data.message,
        progress: data.progress || prev.progress,
        deviceStatuses: {
          ...prev.deviceStatuses,
          ...data.deviceStatuses
        }
      }));
      
      if (data.status === 'completed' || data.status === 'failed') {
        sse.close();
        setIsSyncing(false);
        setEventSource(null);
      }
    };
    
    sse.onerror = () => {
      setSyncStatus(prev => ({
        ...prev,
        status: 'warning',
        message: 'Lost connection to sync updates. The process will continue in the background.'
      }));
      sse.close();
      setIsSyncing(false);
      setEventSource(null);
    };
    
    setEventSource(sse);
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
      case 'queued':
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
            <h1 className="text-2xl font-semibold text-secondary">Device Sync</h1>
            <div className="flex space-x-4">
              <a
                href="/devices"
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
              >
                Back to Devices
              </a>
              <button
                onClick={handleSyncAll}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50"
                disabled={isSyncing || isLoading || devices.length === 0}
              >
                {isSyncing ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Syncing...
                  </>
                ) : (
                  <>
                    <svg className="-ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Sync All Devices
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-6">
          {/* Sync status */}
          {syncStatus.status && (
            <div className={`mb-6 rounded-md p-4 ${getSyncStatusColor(syncStatus.status)}`}>
              <div className="flex">
                <div className="flex-shrink-0">
                  {syncStatus.status === 'completed' && (
                    <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  )}
                  {(syncStatus.status === 'syncing' || syncStatus.status === 'starting') && (
                    <svg className="animate-spin h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  {(syncStatus.status === 'error' || syncStatus.status === 'failed') && (
                    <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  )}
                  {syncStatus.status === 'warning' && (
                    <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                <div className="ml-3 flex-1">
                  <h3 className="text-sm font-medium">
                    {syncStatus.message}
                  </h3>
                  
                  {/* Overall progress bar */}
                  {(syncStatus.status === 'syncing' || syncStatus.status === 'starting') && (
                    <div className="mt-2">
                      <div className="flex justify-between mb-1">
                        <span className="text-xs font-medium">Overall Progress</span>
                        <span className="text-xs font-medium">{syncStatus.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${syncStatus.progress}%` }}></div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

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
                devices.map((device) => {
                  const deviceSyncStatus = syncStatus.deviceStatuses[device.id];
                  
                  return (
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
                                {deviceSyncStatus && (
                                  <span className={`ml-2 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSyncStatusColor(deviceSyncStatus.status)}`}>
                                    {deviceSyncStatus.status}
                                  </span>
                                )}
                              </div>
                              <p className="text-sm text-gray-500">
                                ID: {device.id}
                              </p>
                            </div>
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
                        
                        {/* Device sync status */}
                        {deviceSyncStatus && (
                          <div className="mt-4">
                            <div className={`rounded-md p-3 ${getSyncStatusColor(deviceSyncStatus.status)}`}>
                              <div className="flex">
                                <div className="ml-3">
                                  <p className="text-sm font-medium">
                                    {deviceSyncStatus.message}
                                  </p>
                                  
                                  {/* Progress bar */}
                                  {(deviceSyncStatus.status === 'syncing' || deviceSyncStatus.status === 'starting' || deviceSyncStatus.status === 'queued') && deviceSyncStatus.progress !== undefined && (
                                    <div className="mt-2">
                                      <div className="flex justify-between mb-1">
                                        <span className="text-xs font-medium">Progress</span>
                                        <span className="text-xs font-medium">{deviceSyncStatus.progress}%</span>
                                      </div>
                                      <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${deviceSyncStatus.progress}%` }}></div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Details */}
                                  {deviceSyncStatus.details && (
                                    <div className="mt-2 text-xs">
                                      <p>Files synced: {deviceSyncStatus.details.files_synced}</p>
                                      <p>Total size: {deviceSyncStatus.details.total_size}</p>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </li>
                  );
                })
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

export default DeviceSyncPage;
