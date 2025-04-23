import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../../components/Layout';
import { useChat } from '../../contexts/ChatContext';
import { useLanguage } from '../../contexts/LanguageContext';

const ChatInterface = () => {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const { 
    sessions, 
    currentSession, 
    messages, 
    isLoading, 
    isSending, 
    fetchSessions, 
    createSession, 
    selectSession, 
    sendMessage, 
    sendVoiceMessage 
  } = useChat();
  const { currentLanguage, translate } = useLanguage();
  const router = useRouter();
  const { sessionId } = router.query;
  const messagesEndRef = React.useRef(null);

  // Fetch sessions on mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Select session from URL if provided
  useEffect(() => {
    if (sessionId && sessions.length > 0) {
      const session = sessions.find(s => s.id === sessionId);
      if (session) {
        selectSession(session.id);
      } else {
        // Session not found, redirect to chat list
        router.push('/chat');
      }
    }
  }, [sessionId, sessions]);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Set up media recorder for voice messages
  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          const recorder = new MediaRecorder(stream);
          
          recorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
              setAudioChunks(prev => [...prev, e.data]);
            }
          };
          
          recorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await handleVoiceMessageSend(audioBlob);
            setAudioChunks([]);
          };
          
          setMediaRecorder(recorder);
        })
        .catch(err => {
          console.error('Error accessing microphone:', err);
        });
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleNewSession = async () => {
    await createSession();
  };

  const handleSessionSelect = (id) => {
    router.push(`/chat/${id}`);
  };

  const handleMessageSend = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;
    
    const success = await sendMessage(message, currentLanguage);
    if (success) {
      setMessage('');
    }
  };

  const startRecording = () => {
    if (mediaRecorder) {
      setIsRecording(true);
      mediaRecorder.start();
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      setIsRecording(false);
      mediaRecorder.stop();
    }
  };

  const handleVoiceMessageSend = async (audioBlob) => {
    await sendVoiceMessage(audioBlob, currentLanguage);
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Layout>
      <div className="flex h-[calc(100vh-64px)]">
        {/* Sidebar */}
        <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <button
              onClick={handleNewSession}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            >
              New Chat
            </button>
          </div>
          <div className="flex-1 overflow-y-auto">
            <ul className="divide-y divide-gray-200">
              {sessions.map((session) => (
                <li 
                  key={session.id}
                  className={`cursor-pointer hover:bg-gray-50 ${currentSession?.id === session.id ? 'bg-gray-100' : ''}`}
                  onClick={() => handleSessionSelect(session.id)}
                >
                  <div className="px-4 py-4">
                    <div className="flex justify-between">
                      <p className="text-sm font-medium text-secondary truncate">
                        {session.title || `Chat ${new Date(session.created_at).toLocaleDateString()}`}
                      </p>
                      <p className="text-xs text-gray-500">
                        {new Date(session.updated_at).toLocaleDateString()}
                      </p>
                    </div>
                    <p className="mt-1 text-xs text-gray-500 truncate">
                      {session.last_message || 'No messages yet'}
                    </p>
                  </div>
                </li>
              ))}
              {sessions.length === 0 && !isLoading && (
                <li className="px-4 py-4 text-sm text-gray-500 text-center">
                  No chat sessions yet
                </li>
              )}
              {isLoading && (
                <li className="px-4 py-4 text-sm text-gray-500 text-center">
                  Loading...
                </li>
              )}
            </ul>
          </div>
        </div>

        {/* Chat area */}
        <div className="flex-1 flex flex-col bg-gray-50">
          {currentSession ? (
            <>
              {/* Chat header */}
              <div className="bg-white border-b border-gray-200 p-4">
                <h2 className="text-lg font-medium text-secondary">
                  {currentSession.title || `Chat ${new Date(currentSession.created_at).toLocaleDateString()}`}
                </h2>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg) => (
                  <div 
                    key={msg.id}
                    className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    <div 
                      className={`max-w-3/4 rounded-lg px-4 py-2 ${
                        msg.isUser 
                          ? 'bg-primary text-white' 
                          : 'bg-white border border-gray-200'
                      }`}
                    >
                      <div className="text-sm">
                        {msg.content}
                      </div>
                      <div className={`text-xs mt-1 ${msg.isUser ? 'text-primary-light' : 'text-gray-500'}`}>
                        {formatTime(msg.timestamp)}
                        {msg.language && msg.language !== 'en' && (
                          <span className="ml-2">
                            ({msg.language})
                          </span>
                        )}
                      </div>
                      
                      {/* Sources */}
                      {!msg.isUser && msg.sources && msg.sources.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-gray-200">
                          <p className="text-xs text-gray-500 font-medium">Sources:</p>
                          <ul className="mt-1 text-xs text-gray-500">
                            {msg.sources.map((source, index) => (
                              <li key={index} className="truncate">
                                {source.title || source.id}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {messages.length === 0 && !isLoading && (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <svg className="mx-auto h-12 w-12 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                      </svg>
                      <h3 className="mt-2 text-sm font-medium text-gray-900">No messages</h3>
                      <p className="mt-1 text-sm text-gray-500">
                        Start a conversation by sending a message.
                      </p>
                    </div>
                  </div>
                )}
                {isLoading && (
                  <div className="flex justify-center">
                    <svg className="animate-spin h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input area */}
              <div className="bg-white border-t border-gray-200 p-4">
                <form onSubmit={handleMessageSend} className="flex items-center">
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 appearance-none border rounded-l-md py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-primary focus:border-primary"
                    disabled={isSending}
                  />
                  <button
                    type="button"
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    onTouchStart={startRecording}
                    onTouchEnd={stopRecording}
                    className={`px-4 py-2 border border-l-0 ${
                      isRecording 
                        ? 'bg-red-500 text-white' 
                        : 'bg-gray-100 text-gray-700'
                    }`}
                    disabled={isSending}
                  >
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 border border-l-0 rounded-r-md bg-primary text-white hover:bg-primary-dark disabled:opacity-50"
                    disabled={!message.trim() || isSending}
                  >
                    {isSending ? (
                      <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    )}
                  </button>
                </form>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <svg className="mx-auto h-12 w-12 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <h3 className="mt-2 text-sm font-medium text-gray-900">No chat selected</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Select a chat from the sidebar or start a new one.
                </p>
                <div className="mt-6">
                  <button
                    type="button"
                    onClick={handleNewSession}
                    className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                  >
                    <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                    </svg>
                    New Chat
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default ChatInterface;
