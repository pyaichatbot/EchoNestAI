import React, { useState } from 'react';
import Layout from '../../components/Layout';
import { useDevice } from '../../contexts/DeviceContext';
import { useLanguage } from '../../contexts/LanguageContext';

const RegisterDevicePage = () => {
  const { registerDevice } = useDevice();
  const { translate } = useLanguage();
  const [deviceName, setDeviceName] = useState('');
  const [deviceType, setDeviceType] = useState('tablet');
  const [storageLimit, setStorageLimit] = useState(5);
  const [isRegistering, setIsRegistering] = useState(false);
  const [registrationResult, setRegistrationResult] = useState(null);
  const [qrCode, setQrCode] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!deviceName.trim()) {
      setRegistrationResult({
        success: false,
        message: 'Please provide a device name'
      });
      return;
    }
    
    setIsRegistering(true);
    setRegistrationResult(null);
    
    try {
      const result = await registerDevice({
        name: deviceName,
        type: deviceType,
        storage_limit_gb: storageLimit
      });
      
      if (result.success) {
        setRegistrationResult({
          success: true,
          message: 'Device registered successfully!',
          deviceId: result.deviceId,
          activationCode: result.activationCode
        });
        
        if (result.qrCode) {
          setQrCode(result.qrCode);
        }
      } else {
        setRegistrationResult({
          success: false,
          message: result.message || 'Failed to register device'
        });
      }
    } catch (err) {
      console.error('Error registering device:', err);
      setRegistrationResult({
        success: false,
        message: err.message || 'An error occurred'
      });
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <Layout>
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-semibold text-secondary">Register New Device</h1>
            <a
              href="/devices"
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            >
              Back to Devices
            </a>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-6">
          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              {registrationResult && registrationResult.success ? (
                <div className="text-center">
                  <svg className="mx-auto h-12 w-12 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 className="mt-2 text-lg font-medium text-gray-900">Device Registered Successfully!</h3>
                  <div className="mt-4">
                    <p className="text-sm text-gray-500">
                      Your device has been registered with the following details:
                    </p>
                    <div className="mt-4 bg-gray-50 p-4 rounded-md">
                      <div className="flex justify-between py-2 border-b border-gray-200">
                        <span className="font-medium">Device Name:</span>
                        <span>{deviceName}</span>
                      </div>
                      <div className="flex justify-between py-2 border-b border-gray-200">
                        <span className="font-medium">Device ID:</span>
                        <span className="font-mono">{registrationResult.deviceId}</span>
                      </div>
                      <div className="flex justify-between py-2">
                        <span className="font-medium">Activation Code:</span>
                        <span className="font-mono">{registrationResult.activationCode}</span>
                      </div>
                    </div>
                    
                    {qrCode && (
                      <div className="mt-6">
                        <p className="text-sm text-gray-500 mb-2">
                          Scan this QR code on your device to activate:
                        </p>
                        <div className="flex justify-center">
                          <img src={qrCode} alt="Activation QR Code" className="h-64 w-64" />
                        </div>
                      </div>
                    )}
                    
                    <div className="mt-6">
                      <p className="text-sm text-gray-500">
                        To activate your device, install the EchoNest AI app and enter the activation code or scan the QR code above.
                      </p>
                    </div>
                    
                    <div className="mt-6">
                      <button
                        type="button"
                        onClick={() => {
                          setDeviceName('');
                          setDeviceType('tablet');
                          setStorageLimit(5);
                          setRegistrationResult(null);
                          setQrCode(null);
                        }}
                        className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                      >
                        Register Another Device
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <form onSubmit={handleSubmit}>
                  <div className="space-y-6">
                    {/* Device name */}
                    <div>
                      <label htmlFor="device-name" className="block text-sm font-medium text-gray-700">
                        Device Name
                      </label>
                      <div className="mt-1">
                        <input
                          type="text"
                          name="device-name"
                          id="device-name"
                          value={deviceName}
                          onChange={(e) => setDeviceName(e.target.value)}
                          className="shadow-sm focus:ring-primary focus:border-primary block w-full sm:text-sm border-gray-300 rounded-md"
                          placeholder="Living Room Tablet"
                          required
                        />
                      </div>
                      <p className="mt-1 text-sm text-gray-500">
                        A descriptive name to identify this device.
                      </p>
                    </div>

                    {/* Device type */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700">
                        Device Type
                      </label>
                      <div className="mt-1">
                        <select
                          value={deviceType}
                          onChange={(e) => setDeviceType(e.target.value)}
                          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary focus:border-primary sm:text-sm rounded-md"
                        >
                          <option value="tablet">Tablet</option>
                          <option value="smartphone">Smartphone</option>
                          <option value="desktop">Desktop Computer</option>
                          <option value="laptop">Laptop</option>
                          <option value="other">Other</option>
                        </select>
                      </div>
                    </div>

                    {/* Storage limit */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700">
                        Storage Limit (GB)
                      </label>
                      <div className="mt-1">
                        <input
                          type="range"
                          min="1"
                          max="20"
                          value={storageLimit}
                          onChange={(e) => setStorageLimit(parseInt(e.target.value))}
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-gray-500 px-1">
                          <span>1 GB</span>
                          <span>{storageLimit} GB</span>
                          <span>20 GB</span>
                        </div>
                      </div>
                      <p className="mt-1 text-sm text-gray-500">
                        Maximum storage space this device can use for offline content.
                      </p>
                    </div>

                    {/* Registration error */}
                    {registrationResult && !registrationResult.success && (
                      <div className="rounded-md bg-red-50 p-4">
                        <div className="flex">
                          <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="ml-3">
                            <h3 className="text-sm font-medium text-red-800">
                              {registrationResult.message}
                            </h3>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Submit button */}
                    <div className="flex justify-end">
                      <button
                        type="submit"
                        className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50"
                        disabled={isRegistering}
                      >
                        {isRegistering ? (
                          <>
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Registering...
                          </>
                        ) : (
                          'Register Device'
                        )}
                      </button>
                    </div>
                  </div>
                </form>
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default RegisterDevicePage;
