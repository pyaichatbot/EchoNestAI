import React, { useState, useEffect } from 'react';
import Layout from '../../components/Layout';
import { useContent } from '../../contexts/ContentContext';
import { useLanguage } from '../../contexts/LanguageContext';

const ContentPage = () => {
  const { contents, fetchContents, deleteContent, assignContent } = useContent();
  const { translate } = useLanguage();
  const [selectedContent, setSelectedContent] = useState(null);
  const [isAssignModalOpen, setIsAssignModalOpen] = useState(false);
  const [assignData, setAssignData] = useState({
    childIds: [],
    groupIds: [],
    deviceIds: []
  });
  const [children, setChildren] = useState([]);
  const [groups, setGroups] = useState([]);
  const [devices, setDevices] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch content on mount
  useEffect(() => {
    fetchContents();
    fetchAssignmentOptions();
  }, []);

  // Fetch children, groups, and devices for assignment
  const fetchAssignmentOptions = async () => {
    setIsLoading(true);
    try {
      const token = localStorage.getItem('token');
      
      // Fetch children
      const childrenResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/children`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (childrenResponse.ok) {
        const childrenData = await childrenResponse.json();
        setChildren(childrenData);
      }
      
      // Fetch groups
      const groupsResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/groups`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (groupsResponse.ok) {
        const groupsData = await groupsResponse.json();
        setGroups(groupsData);
      }
      
      // Fetch devices
      const devicesResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/devices`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (devicesResponse.ok) {
        const devicesData = await devicesResponse.json();
        setDevices(devicesData);
      }
    } catch (err) {
      console.error('Error fetching assignment options:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle content deletion
  const handleDelete = async (contentId) => {
    if (window.confirm('Are you sure you want to delete this content?')) {
      await deleteContent(contentId);
    }
  };

  // Open assign modal
  const handleOpenAssignModal = (content) => {
    setSelectedContent(content);
    
    // Pre-populate with existing assignments
    setAssignData({
      childIds: content.assigned_children?.map(child => child.id) || [],
      groupIds: content.assigned_groups?.map(group => group.id) || [],
      deviceIds: content.assigned_devices?.map(device => device.id) || []
    });
    
    setIsAssignModalOpen(true);
  };

  // Handle assignment changes
  const handleAssignmentChange = (type, id, checked) => {
    setAssignData(prev => {
      const key = `${type}Ids`;
      if (checked) {
        return {
          ...prev,
          [key]: [...prev[key], id]
        };
      } else {
        return {
          ...prev,
          [key]: prev[key].filter(itemId => itemId !== id)
        };
      }
    });
  };

  // Submit assignment
  const handleAssignSubmit = async () => {
    if (selectedContent) {
      await assignContent(selectedContent.id, assignData);
      setIsAssignModalOpen(false);
    }
  };

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString();
  };

  // Get content type icon
  const getContentTypeIcon = (type) => {
    switch (type) {
      case 'document':
        return (
          <svg className="h-6 w-6 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        );
      case 'audio':
        return (
          <svg className="h-6 w-6 text-green-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
        );
      case 'video':
        return (
          <svg className="h-6 w-6 text-red-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        );
      default:
        return (
          <svg className="h-6 w-6 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        );
    }
  };

  return (
    <Layout>
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-semibold text-secondary">Content Library</h1>
            <a
              href="/content/upload"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            >
              Upload Content
            </a>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-6">
          {/* Content list */}
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <ul className="divide-y divide-gray-200">
              {contents.length > 0 ? (
                contents.map((content) => (
                  <li key={content.id}>
                    <div className="px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="flex-shrink-0">
                            {getContentTypeIcon(content.type)}
                          </div>
                          <div className="ml-4">
                            <p className="text-sm font-medium text-primary truncate">
                              {content.title}
                            </p>
                            <p className="text-sm text-gray-500">
                              {content.description || 'No description'}
                            </p>
                          </div>
                        </div>
                        <div className="ml-2 flex-shrink-0 flex">
                          <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            {content.language || 'en'}
                          </p>
                        </div>
                      </div>
                      <div className="mt-2 sm:flex sm:justify-between">
                        <div className="sm:flex">
                          <p className="flex items-center text-sm text-gray-500">
                            <svg className="flex-shrink-0 mr-1.5 h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd" />
                            </svg>
                            Uploaded on {formatDate(content.created_at)}
                          </p>
                          <p className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0 sm:ml-6">
                            <svg className="flex-shrink-0 mr-1.5 h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                            </svg>
                            {content.assigned_children?.length || 0} children, {content.assigned_groups?.length || 0} groups
                          </p>
                        </div>
                        <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                          <button
                            onClick={() => handleOpenAssignModal(content)}
                            className="mr-2 inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-primary-dark bg-primary-light bg-opacity-20 hover:bg-opacity-30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                          >
                            Assign
                          </button>
                          <button
                            onClick={() => handleDelete(content.id)}
                            className="inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-red-700 bg-red-100 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  </li>
                ))
              ) : (
                <li className="px-4 py-5 sm:px-6 text-center text-gray-500">
                  No content available. Upload some content to get started!
                </li>
              )}
            </ul>
          </div>
        </div>
      </div>

      {/* Assignment Modal */}
      {isAssignModalOpen && selectedContent && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                    <h3 className="text-lg leading-6 font-medium text-secondary">
                      Assign Content: {selectedContent.title}
                    </h3>
                    <div className="mt-4 max-h-96 overflow-y-auto">
                      {/* Children */}
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Children</h4>
                        {children.length > 0 ? (
                          <div className="space-y-2">
                            {children.map(child => (
                              <div key={child.id} className="flex items-center">
                                <input
                                  id={`child-${child.id}`}
                                  type="checkbox"
                                  className="h-4 w-4 text-primary focus:ring-primary border-gray-300 rounded"
                                  checked={assignData.childIds.includes(child.id)}
                                  onChange={(e) => handleAssignmentChange('child', child.id, e.target.checked)}
                                />
                                <label htmlFor={`child-${child.id}`} className="ml-2 block text-sm text-gray-900">
                                  {child.name}
                                </label>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-gray-500">No children available</p>
                        )}
                      </div>

                      {/* Groups */}
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Groups</h4>
                        {groups.length > 0 ? (
                          <div className="space-y-2">
                            {groups.map(group => (
                              <div key={group.id} className="flex items-center">
                                <input
                                  id={`group-${group.id}`}
                                  type="checkbox"
                                  className="h-4 w-4 text-primary focus:ring-primary border-gray-300 rounded"
                                  checked={assignData.groupIds.includes(group.id)}
                                  onChange={(e) => handleAssignmentChange('group', group.id, e.target.checked)}
                                />
                                <label htmlFor={`group-${group.id}`} className="ml-2 block text-sm text-gray-900">
                                  {group.name}
                                </label>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-gray-500">No groups available</p>
                        )}
                      </div>

                      {/* Devices */}
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Devices</h4>
                        {devices.length > 0 ? (
                          <div className="space-y-2">
                            {devices.map(device => (
                              <div key={device.id} className="flex items-center">
                                <input
                                  id={`device-${device.id}`}
                                  type="checkbox"
                                  className="h-4 w-4 text-primary focus:ring-primary border-gray-300 rounded"
                                  checked={assignData.deviceIds.includes(device.id)}
                                  onChange={(e) => handleAssignmentChange('device', device.id, e.target.checked)}
                                />
                                <label htmlFor={`device-${device.id}`} className="ml-2 block text-sm text-gray-900">
                                  {device.name}
                                </label>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-gray-500">No devices available</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button
                  type="button"
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-primary text-base font-medium text-white hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={handleAssignSubmit}
                >
                  Assign
                </button>
                <button
                  type="button"
                  className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={() => setIsAssignModalOpen(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Layout>
  );
};

export default ContentPage;
