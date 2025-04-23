import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

// Device context for managing device functionality
export const DeviceContext = React.createContext({
  devices: [],
  selectedDevice: null,
  syncStatus: null,
  isLoading: false,
  error: null,
  fetchDevices: () => {},
  selectDevice: () => {},
  registerDevice: () => {},
  syncDevice: () => {},
  getSyncStatus: () => {},
});

export const DeviceProvider = ({ children }) => {
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [syncStatus, setSyncStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const { isAuthenticated } = useAuth();

  // Fetch devices on mount if authenticated
  useEffect(() => {
    if (isAuthenticated) {
      fetchDevices();
    }
  }, [isAuthenticated]);

  // Fetch devices
  const fetchDevices = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setDevices(data);
        
        // Select first device if none selected
        if (data.length > 0 && !selectedDevice) {
          selectDevice(data[0].id);
        }
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch devices');
      }
    } catch (err) {
      console.error('Fetch devices error:', err);
      setError('Failed to fetch devices');
    } finally {
      setIsLoading(false);
    }
  };

  // Select device
  const selectDevice = (deviceId) => {
    const device = devices.find(d => d.id === deviceId);
    if (device) {
      setSelectedDevice(device);
      getSyncStatus(deviceId);
    }
  };

  // Register new device
  const registerDevice = async (name) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices/register`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ name })
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        
        // Add new device to list
        setDevices([...devices, data]);
        
        // Select the new device
        setSelectedDevice(data);
        return true;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to register device');
        return false;
      }
    } catch (err) {
      console.error('Register device error:', err);
      setError('Failed to register device');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Sync device
  const syncDevice = async (deviceId) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices/${deviceId}/sync`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({})
        }
      );
      
      if (response.ok) {
        // Update sync status
        await getSyncStatus(deviceId);
        return true;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to sync device');
        return false;
      }
    } catch (err) {
      console.error('Sync device error:', err);
      setError('Failed to sync device');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Get sync status
  const getSyncStatus = async (deviceId) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices/${deviceId}/sync-status`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setSyncStatus(data);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to get sync status');
      }
    } catch (err) {
      console.error('Get sync status error:', err);
      setError('Failed to get sync status');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DeviceContext.Provider
      value={{
        devices,
        selectedDevice,
        syncStatus,
        isLoading,
        error,
        fetchDevices,
        selectDevice,
        registerDevice,
        syncDevice,
        getSyncStatus,
      }}
    >
      {children}
    </DeviceContext.Provider>
  );
};

// Custom hook to use device context
export const useDevice = () => {
  const context = React.useContext(DeviceContext);
  if (context === undefined) {
    throw new Error('useDevice must be used within a DeviceProvider');
  }
  return context;
};
