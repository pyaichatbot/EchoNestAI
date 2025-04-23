import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

// Chat context for managing chat functionality
export const ChatContext = React.createContext({
  sessions: [],
  currentSession: null,
  messages: [],
  isLoading: false,
  isSending: false,
  error: null,
  fetchSessions: () => {},
  createSession: () => {},
  selectSession: () => {},
  fetchMessages: () => {},
  sendMessage: () => {},
  sendVoiceMessage: () => {},
});

export const ChatProvider = ({ children }) => {
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const [eventSource, setEventSource] = useState(null);
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
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${currentSession.id}/stream?token=${token}`
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
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/sessions`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setSessions(data);
        
        // Select first session if none selected
        if (data.length > 0 && !currentSession) {
          await selectSession(data[0].id);
        }
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch chat sessions');
      }
    } catch (err) {
      console.error('Fetch sessions error:', err);
      setError('Failed to fetch chat sessions');
    } finally {
      setIsLoading(false);
    }
  };

  // Create new chat session
  const createSession = async (title) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/sessions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          title: title || `Chat ${new Date().toLocaleString()}` 
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        // Add new session to list
        setSessions([...sessions, data]);
        
        // Select the new session
        await selectSession(data.id);
        
        return data.id;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to create chat session');
        return null;
      }
    } catch (err) {
      console.error('Create session error:', err);
      setError('Failed to create chat session');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  // Select chat session
  const selectSession = async (sessionId) => {
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
      } else {
        setError('Session not found');
      }
    } catch (err) {
      console.error('Select session error:', err);
      setError('Failed to select chat session');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch messages for a session
  const fetchMessages = async (sessionId) => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}/messages`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        
        // Transform messages to our format
        const formattedMessages = data.map((msg) => ({
          id: msg.id,
          content: msg.content,
          isUser: msg.is_user,
          timestamp: msg.timestamp,
          language: msg.language,
          sources: msg.sources,
        }));
        
        setMessages(formattedMessages);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch chat messages');
      }
    } catch (err) {
      console.error('Fetch messages error:', err);
      setError('Failed to fetch chat messages');
    } finally {
      setIsLoading(false);
    }
  };

  // Send text message
  const sendMessage = async (content, language) => {
    if (!currentSession) {
      setError('No active chat session');
      return false;
    }
    
    setIsSending(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${currentSession.id}/messages`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            content,
            language: language || 'en',
          })
        }
      );
      
      if (response.ok) {
        // Message will be added via SSE
        return true;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to send message');
        return false;
      }
    } catch (err) {
      console.error('Send message error:', err);
      setError('Failed to send message');
      return false;
    } finally {
      setIsSending(false);
    }
  };

  // Send voice message
  const sendVoiceMessage = async (audioBlob, language) => {
    if (!currentSession) {
      setError('No active chat session');
      return false;
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
      
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${currentSession.id}/voice`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
          body: formData
        }
      );
      
      if (response.ok) {
        // Message will be added via SSE
        return true;
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to send voice message');
        return false;
      }
    } catch (err) {
      console.error('Send voice message error:', err);
      setError('Failed to send voice message');
      return false;
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
  const context = React.useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
