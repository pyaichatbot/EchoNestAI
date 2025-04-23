import { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

// Define types
interface ChatMessage {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  language?: string;
  sources?: string[];
}

interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message?: string;
}

interface ChatContextType {
  sessions: ChatSession[];
  currentSession: ChatSession | null;
  messages: ChatMessage[];
  isLoading: boolean;
  isSending: boolean;
  error: string | null;
  fetchSessions: () => Promise<void>;
  createSession: (title?: string) => Promise<string>;
  selectSession: (sessionId: string) => Promise<void>;
  fetchMessages: (sessionId: string) => Promise<void>;
  sendMessage: (content: string, language?: string) => Promise<void>;
  sendVoiceMessage: (audioBlob: Blob, language?: string) => Promise<void>;
}

interface ChatProviderProps {
  children: ReactNode;
}

// Create context
const ChatContext = createContext<ChatContextType | undefined>(undefined);

// Create provider
export const ChatProvider = ({ children }: ChatProviderProps) => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSending, setIsSending] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const { isAuthenticated } = useAuth();

  // Fetch sessions on mount if authenticated
  useEffect(() => {
    if (isAuthenticated) {
      fetchSessions();
    }
  }, [isAuthenticated]);

  // Set up SSE for chat messages
  useEffect(() => {
    if (isAuthenticated && currentSession) {
      const token = localStorage.getItem('token');
      const sse = new EventSource(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/chat/${currentSession.id}/stream?token=${token}`
      );

      sse.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'chat_message') {
          // Add new message to the list
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              id: data.id,
              content: data.content,
              isUser: data.is_user,
              timestamp: data.timestamp,
              language: data.language,
              sources: data.sources,
            },
          ]);
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
  }, [isAuthenticated, currentSession]);

  // Fetch chat sessions
  const fetchSessions = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${process.env.API_URL}/api/${process.env.API_VERSION}/chat/sessions`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setSessions(response.data);
      
      // Select first session if none selected
      if (response.data.length > 0 && !currentSession) {
        await selectSession(response.data[0].id);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch chat sessions');
    } finally {
      setIsLoading(false);
    }
  };

  // Create new chat session
  const createSession = async (title?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/chat/sessions`,
        { title: title || `Chat ${new Date().toLocaleString()}` },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Add new session to list
      setSessions([...sessions, response.data]);
      
      // Select the new session
      await selectSession(response.data.id);
      
      return response.data.id;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create chat session');
      return '';
    } finally {
      setIsLoading(false);
    }
  };

  // Select chat session
  const selectSession = async (sessionId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      // Find session in list
      const session = sessions.find(s => s.id === sessionId);
      if (session) {
        setCurrentSession(session);
        
        // Close existing event source
        if (eventSource) {
          eventSource.close();
          setEventSource(null);
        }
        
        // Fetch messages for this session
        await fetchMessages(sessionId);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to select chat session');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch messages for a session
  const fetchMessages = async (sessionId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/chat/${sessionId}/messages`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Transform messages to our format
      const formattedMessages = response.data.map((msg: any) => ({
        id: msg.id,
        content: msg.content,
        isUser: msg.is_user,
        timestamp: msg.timestamp,
        language: msg.language,
        sources: msg.sources,
      }));
      
      setMessages(formattedMessages);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch chat messages');
    } finally {
      setIsLoading(false);
    }
  };

  // Send text message
  const sendMessage = async (content: string, language?: string) => {
    if (!currentSession) {
      setError('No active chat session');
      return;
    }
    
    setIsSending(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/chat/${currentSession.id}/messages`,
        {
          content,
          language: language || 'en',
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Message will be added via SSE
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to send message');
    } finally {
      setIsSending(false);
    }
  };

  // Send voice message
  const sendVoiceMessage = async (audioBlob: Blob, language?: string) => {
    if (!currentSession) {
      setError('No active chat session');
      return;
    }
    
    setIsSending(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      
      // Create form data
      const formData = new FormData();
      formData.append('audio', audioBlob);
      if (language) {
        formData.append('language', language);
      }
      
      await axios.post(
        `${process.env.API_URL}/api/${process.env.API_VERSION}/chat/${currentSession.id}/voice`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      // Message will be added via SSE
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to send voice message');
    } finally {
      setIsSending(false);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        sessions,
        currentSession,
        messages,
        isLoading,
        isSending,
        error,
        fetchSessions,
        createSession,
        selectSession,
        fetchMessages,
        sendMessage,
        sendVoiceMessage,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook to use chat context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
