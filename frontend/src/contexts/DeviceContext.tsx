import { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

// Define types
interface Device {
  id: string;
  name: string;
  status: string;
  lastSync: string;
  mode: string;
}

interface SyncStatus {
  deviceId: string;
  syncRequired: boolean;
  lastSyncTime: string;
  pendingDocuments: number;
}

interface DeviceContextType {
  devices: Device[];
  selectedDevice: Device | null;
  syncStatus: SyncStatus | null;
  isLoading: boolean;
  error: string | null;
  fetchDevices: () => Promise<void>;
  selectDevice: (deviceId: string) => void;
  registerDevice: (name: string) => Promise<void>;
  syncDevice: (deviceId: string) => Promise<void>;
  getSyncStatus: (deviceId: string) => Promise<void>;
}

interface DeviceProviderProps {
  children: ReactNode;
}

// Create context
const DeviceContext = createContext<DeviceContextType | undefined>(undefined);

// Create provider
export const DeviceProvider = ({ children }: DeviceProviderProps) => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
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
      const response = await axios.get(`${process.env.API_URL}/api/${process.env.API_VERSION}/devices`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setDevices(response.data);
      
      // Select first device if none selected
      if (response.data.length > 0 && !selectedDevice) {
        setSelectedDevice(response.data[0]);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch devices');
    } finally {
      setIsLoading(false);
    }
  };

  // Select device
  const selectDevice = (deviceId: string) => {
    const device = devices.find(d => d.id === deviceId);
    if (device) {
      setSelectedDevice(device);
      getSyncStatus(deviceId);
    }
  };

  // Register new device
  const registerDevice = async (name: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/devices/register`,
        { name },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Add new device to list
      setDevices([...devices, response.data]);
      
      // Select the new device
      setSelectedDevice(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to register device');
    } finally {
      setIsLoading(false);
    }
  };

  // Sync device
  const syncDevice = async (deviceId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/devices/${deviceId}/sync`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Update sync status
      await getSyncStatus(deviceId);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to sync device');
    } finally {
      setIsLoading(false);
    }
  };

  // Get sync status
  const getSyncStatus = async (deviceId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/devices/${deviceId}/sync-status`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      setSyncStatus(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get sync status');
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
  const context = useContext(DeviceContext);
  if (context === undefined) {
    throw new Error('useDevice must be used within a DeviceProvider');
  }
  return context;
};
