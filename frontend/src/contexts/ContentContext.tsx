import { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

// Define types
interface Content {
  id: string;
  title: string;
  type: string;
  language: string;
  size_mb: number;
  status: string;
  created_at: string;
  updated_at: string;
}

interface ContentUploadProgress {
  contentId: string;
  progress: number;
  status: string;
  message?: string;
}

interface ContentContextType {
  contents: Content[];
  uploadProgress: ContentUploadProgress | null;
  isLoading: boolean;
  error: string | null;
  fetchContents: () => Promise<void>;
  uploadContent: (file: File, metadata: ContentMetadata) => Promise<void>;
  deleteContent: (contentId: string) => Promise<void>;
  assignContent: (contentId: string, assignTo: AssignmentData) => Promise<void>;
}

interface ContentMetadata {
  title: string;
  description: string;
  language: string;
  tags?: string[];
}

interface AssignmentData {
  childIds?: string[];
  groupIds?: string[];
}

interface ContentProviderProps {
  children: ReactNode;
}

// Create context
const ContentContext = createContext<ContentContextType | undefined>(undefined);

// Create provider
export const ContentProvider = ({ children }: ContentProviderProps) => {
  const [contents, setContents] = useState<Content[]>([]);
  const [uploadProgress, setUploadProgress] = useState<ContentUploadProgress | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const { isAuthenticated } = useAuth();
  const [eventSource, setEventSource] = useState<EventSource | null>(null);

  // Set up SSE for content upload progress
  useEffect(() => {
    if (isAuthenticated) {
      const token = localStorage.getItem('token');
      const sse = new EventSource(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/events/content-progress?token=${token}`
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
      const response = await axios.get(`${process.env.API_URL}/api/${process.env.API_VERSION}/content`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setContents(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch contents');
    } finally {
      setIsLoading(false);
    }
  };

  // Upload content
  const uploadContent = async (file: File, metadata: ContentMetadata) => {
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
      
      // Upload file
      const response = await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/content/upload`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!);
            setUploadProgress({
              contentId: 'uploading',
              progress: percentCompleted,
              status: 'uploading',
            });
          },
        }
      );
      
      // Set initial progress for processing
      setUploadProgress({
        contentId: response.data.id,
        progress: 0,
        status: 'processing',
      });
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload content');
      setUploadProgress(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Delete content
  const deleteContent = async (contentId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`${process.env.API_URL}/api/${process.env.API_VERSION}/content/${contentId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      // Update content list
      setContents(contents.filter(content => content.id !== contentId));
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete content');
    } finally {
      setIsLoading(false);
    }
  };

  // Assign content
  const assignContent = async (contentId: string, assignTo: AssignmentData) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/content/${contentId}/assign`,
        assignTo,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Refresh content list to get updated assignments
      await fetchContents();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to assign content');
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
  const context = useContext(ContentContext);
  if (context === undefined) {
    throw new Error('useContent must be used within a ContentProvider');
  }
  return context;
};
