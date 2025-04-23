import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

// Content context for managing content functionality
export const ContentContext = React.createContext({
  contents: [],
  uploadProgress: null,
  isLoading: false,
  error: null,
  fetchContents: () => {},
  uploadContent: () => {},
  deleteContent: () => {},
  assignContent: () => {},
});

export const ContentProvider = ({ children }) => {
  const [contents, setContents] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [eventSource, setEventSource] = useState(null);
  const { isAuthenticated } = useAuth();

  // Set up SSE for content upload progress
  useEffect(() => {
    if (isAuthenticated) {
      const token = localStorage.getItem('token');
      const sse = new EventSource(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/events/content-progress?token=${token}`
      );

      sse.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'content_progress') {
          setUploadProgress({
            contentId: data.content_id,
            progress: data.progress,
            status: data.status,
            message: data.message,
          });

          // If content is processed, refresh content list
          if (data.status === 'completed') {
            fetchContents();
          }
        }
      };

      sse.onerror = () => {
        sse.close();
      };

      setEventSource(sse);

      return () => {
        sse.close();
      };
    }
  }, [isAuthenticated]);

  // Fetch contents on mount if authenticated
  useEffect(() => {
    if (isAuthenticated) {
      fetchContents();
    }
  }, [isAuthenticated]);

  // Fetch contents
  const fetchContents = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/content`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setContents(data);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch contents');
      }
    } catch (err) {
      console.error('Fetch contents error:', err);
      setError('Failed to fetch contents');
    } finally {
      setIsLoading(false);
    }
  };

  // Upload content
  const uploadContent = async (file, metadata) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', metadata.title);
      formData.append('description', metadata.description);
      formData.append('language', metadata.language);
      
      if (metadata.tags && metadata.tags.length > 0) {
        formData.append('tags', JSON.stringify(metadata.tags));
      }
      
      // Upload file with progress tracking
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${process.env.NEXT_PUBLIC_API_URL}/api/v1/content/upload`, true);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentCompleted = Math.round((event.loaded * 100) / event.total);
          setUploadProgress({
            contentId: 'uploading',
            progress: percentCompleted,
            status: 'uploading',
          });
        }
      };
      
      xhr.onload = function() {
        if (xhr.status === 200 || xhr.status === 201) {
          const response = JSON.parse(xhr.responseText);
          // Set initial progress for processing
          setUploadProgress({
            contentId: response.id,
            progress: 0,
            status: 'processing',
          });
          setIsLoading(false);
        } else {
          let errorMessage = 'Failed to upload content';
          try {
            const errorData = JSON.parse(xhr.responseText);
            errorMessage = errorData.detail || errorMessage;
          } catch (e) {
            // If parsing fails, use default error message
          }
          setError(errorMessage);
          setUploadProgress(null);
          setIsLoading(false);
        }
      };
      
      xhr.onerror = function() {
        setError('Failed to upload content');
        setUploadProgress(null);
        setIsLoading(false);
      };
      
      xhr.send(formData);
      
      return true;
    } catch (err) {
      console.error('Upload content error:', err);
      setError('Failed to upload content');
      setUploadProgress(null);
      setIsLoading(false);
      return false;
    }
  };

  // Delete content
  const deleteContent = async (contentId) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/content/${contentId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        // Update content list
        setContents(contents.filter(content => content.id !== contentId));
        return true;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to delete content');
        return false;
      }
    } catch (err) {
      console.error('Delete content error:', err);
      setError('Failed to delete content');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Assign content
  const assignContent = async (contentId, assignTo) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/content/${contentId}/assign`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(assignTo)
        }
      );
      
      if (response.ok) {
        // Refresh content list to get updated assignments
        await fetchContents();
        return true;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to assign content');
        return false;
      }
    } catch (err) {
      console.error('Assign content error:', err);
      setError('Failed to assign content');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ContentContext.Provider
      value={{
        contents,
        uploadProgress,
        isLoading,
        error,
        fetchContents,
        uploadContent,
        deleteContent,
        assignContent,
      }}
    >
      {children}
    </ContentContext.Provider>
  );
};

// Custom hook to use content context
export const useContent = () => {
  const context = React.useContext(ContentContext);
  if (context === undefined) {
    throw new Error('useContent must be used within a ContentProvider');
  }
  return context;
};
