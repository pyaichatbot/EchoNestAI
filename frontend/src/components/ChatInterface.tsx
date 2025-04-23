import React from 'react';
import { useChat } from '../contexts/ChatContext';
import { useLanguage } from '../contexts/LanguageContext';

const ChatInterface: React.FC = () => {
  const { currentSession, messages, isSending, sendMessage, sendVoiceMessage } = useChat();
  const { currentLanguage } = useLanguage();
  const [inputText, setInputText] = React.useState('');
  const [isRecording, setIsRecording] = React.useState(false);
  const [mediaRecorder, setMediaRecorder] = React.useState<MediaRecorder | null>(null);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle text input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
  };

  // Handle send message
  const handleSendMessage = async () => {
    if (inputText.trim() === '') return;
    
    await sendMessage(inputText, currentLanguage);
    setInputText('');
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Start voice recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const audioChunks: BlobPart[] = [];
      
      recorder.addEventListener('dataavailable', (event) => {
        audioChunks.push(event.data);
      });
      
      recorder.addEventListener('stop', async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await sendVoiceMessage(audioBlob, currentLanguage);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      });
      
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  // Stop voice recording
  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      setMediaRecorder(null);
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (!currentSession) {
    return (
      <div className="flex justify-center items-center h-full">
        <p className="text-base">No active chat session. Please create or select a session.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-120px)]">
      {/* Chat header */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-4">
        <h2 className="text-xl font-medium">{currentSession.title}</h2>
      </div>
      
      {/* Messages container */}
      <div className="flex-grow overflow-y-auto p-4 bg-gray-100 rounded mb-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-4`}
          >
            <div
              className={`p-4 max-w-[70%] rounded-lg shadow-sm ${
                message.isUser ? 'bg-primary-50' : 'bg-white'
              }`}
            >
              <p className="text-base">{message.content}</p>
              <div className="flex justify-between mt-2">
                <span className="text-xs text-gray-500">
                  {message.language && `${message.language.toUpperCase()}`}
                </span>
                <span className="text-xs text-gray-500">
                  {formatTimestamp(message.timestamp)}
                </span>
              </div>
              {message.sources && message.sources.length > 0 && (
                <div className="mt-2">
                  <span className="text-xs text-gray-500">
                    Sources: {message.sources.join(', ')}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <input
            type="text"
            className="w-full border border-gray-300 rounded p-2 mr-2 focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500"
            placeholder="Type your message..."
            value={inputText}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            disabled={isSending || isRecording}
          />
          {isRecording ? (
            <button
              className="bg-secondary-500 hover:bg-secondary-600 text-white rounded p-2 flex items-center"
              onClick={stopRecording}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
              </svg>
              Stop
            </button>
          ) : (
            <button
              className="border border-gray-300 text-gray-700 rounded p-2 mr-2 hover:bg-gray-50 flex items-center disabled:bg-gray-100 disabled:text-gray-500 disabled:border-gray-200"
              onClick={startRecording}
              disabled={isSending}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
              </svg>
              Voice
            </button>
          )}
          <button
            className={`rounded p-2 flex items-center ${
              inputText.trim() === '' || isSending || isRecording
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-primary-500 hover:bg-primary-600 text-white'
            }`}
            onClick={handleSendMessage}
            disabled={inputText.trim() === '' || isSending || isRecording}
          >
            {isSending ? (
              <svg className="animate-spin h-5 w-5 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            )}
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
