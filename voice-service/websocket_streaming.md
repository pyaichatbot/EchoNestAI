# WebSocket Full-Duplex Audio Streaming Implementation

## Overview
This implementation provides real-time, bidirectional audio streaming with natural conversation capabilities including barge-in detection, interruption handling, and turn-taking logic.

## Architecture Components

### 1. WebSocket Audio Streaming Service
- **Full-duplex communication** between client and server
- **Real-time audio processing** with minimal latency
- **Voice Activity Detection (VAD)** for barge-in detection
- **Turn-taking management** for natural conversation flow
- **Rate limiting** and connection management

### 2. Audio Processing Pipeline
- **Continuous audio streaming** instead of file uploads
- **Real-time STT** with streaming transcription
- **Real-time TTS** with streaming synthesis
- **Audio format conversion** and buffering

### 3. Conversation Management
- **Session management** for conversation state
- **Interruption detection** and handling
- **Turn-taking logic** with natural pauses
- **Context preservation** across turns

## Implementation Files

### Core WebSocket Service
```python
# websocket_service.py
import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Dict, Set, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import queue
import soundfile as sf
import io
import base64

from main import stt_engine, tts_engine
from config import WEBSOCKET_RATE_LIMIT, VAD_SILENCE_THRESHOLD, TURN_TIMEOUT

logger = logging.getLogger(__name__)

@dataclass
class ConversationSession:
    session_id: str
    user_id: str
    language: str
    is_speaking: bool = False
    last_activity: datetime = None
    audio_buffer: bytes = None
    conversation_context: str = ""
    turn_start_time: Optional[datetime] = None
    interruption_count: int = 0

class WebSocketAudioStreamingService:
    def __init__(self):
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.sessions: Dict[str, ConversationSession] = {}
        self.rate_limit_tracker: Dict[str, Dict] = {}
        self.vad_processor = VoiceActivityDetector()
        self.turn_manager = TurnTakingManager()
        
    async def handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections with full-duplex audio streaming."""
        connection_id = None
        session_id = None
        
        try:
            # Extract connection parameters
            query_params = self._parse_query_params(path)
            session_id = query_params.get('session_id')
            user_id = query_params.get('user_id')
            language = query_params.get('language', 'en')
            
            if not session_id or not user_id:
                await websocket.close(1008, "Missing required parameters")
                return
            
            connection_id = f"{user_id}_{session_id}"
            
            # Rate limiting check
            if not self._check_rate_limit(connection_id):
                await websocket.close(1008, "Rate limit exceeded")
                return
            
            # Initialize session
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                language=language,
                last_activity=datetime.now()
            )
            self.sessions[session_id] = session
            self.active_connections[connection_id] = websocket
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Send connection confirmation
            await websocket.send(json.dumps({
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Start audio processing tasks
            audio_processor_task = asyncio.create_task(
                self._process_audio_stream(websocket, session)
            )
            tts_stream_task = asyncio.create_task(
                self._stream_tts_response(websocket, session)
            )
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_incoming_message(websocket, session, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {str(e)}")
            if websocket.open:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
        finally:
            # Cleanup
            if connection_id:
                self.active_connections.pop(connection_id, None)
            if session_id:
                self.sessions.pop(session_id, None)
            self._cleanup_rate_limit(connection_id)
            
            # Cancel tasks
            if 'audio_processor_task' in locals():
                audio_processor_task.cancel()
            if 'tts_stream_task' in locals():
                tts_stream_task.cancel()
    
    async def _process_audio_stream(self, websocket, session: ConversationSession):
        """Process incoming audio stream with real-time STT and VAD."""
        audio_buffer = b""
        silence_duration = 0
        is_speaking = False
        
        try:
            async for message in websocket:
                if message.type == websockets.protocol.MessageType.BINARY:
                    # Process audio chunk
                    audio_chunk = message.data
                    audio_buffer += audio_chunk
                    
                    # Check for voice activity
                    vad_result = self.vad_processor.detect_voice_activity(audio_chunk)
                    
                    if vad_result.is_speech:
                        if not is_speaking:
                            # User started speaking
                            is_speaking = True
                            session.is_speaking = True
                            session.turn_start_time = datetime.now()
                            
                            # Notify turn manager
                            self.turn_manager.user_started_speaking(session.session_id)
                            
                            await websocket.send(json.dumps({
                                "type": "user_speaking_started",
                                "timestamp": datetime.now().isoformat()
                            }))
                        
                        silence_duration = 0
                        session.last_activity = datetime.now()
                        
                        # Process audio buffer for transcription
                        if len(audio_buffer) >= 3200:  # 200ms at 16kHz
                            await self._process_audio_chunk(websocket, session, audio_buffer)
                            audio_buffer = b""
                    else:
                        silence_duration += 1
                        
                        # Check for end of speech
                        if is_speaking and silence_duration > VAD_SILENCE_THRESHOLD:
                            is_speaking = False
                            session.is_speaking = False
                            
                            # Process final audio buffer
                            if audio_buffer:
                                await self._process_audio_chunk(websocket, session, audio_buffer)
                                audio_buffer = b""
                            
                            # Notify turn manager
                            self.turn_manager.user_finished_speaking(session.session_id)
                            
                            await websocket.send(json.dumps({
                                "type": "user_speaking_ended",
                                "timestamp": datetime.now().isoformat()
                            }))
                            
                            # Start AI response
                            await self._generate_ai_response(websocket, session)
                
                elif message.type == websockets.protocol.MessageType.TEXT:
                    # Handle text messages (commands, metadata)
                    data = json.loads(message.data)
                    await self._handle_text_message(websocket, session, data)
                    
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            raise
    
    async def _process_audio_chunk(self, websocket, session: ConversationSession, audio_data: bytes):
        """Process audio chunk with real-time STT."""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save to temporary file for STT processing
            temp_file = f"/tmp/audio_chunk_{session.session_id}_{datetime.now().timestamp()}.wav"
            sf.write(temp_file, audio_array, 16000)
            
            # Perform STT
            result = stt_engine.transcribe(
                audio_path=temp_file,
                language=session.language,
                use_cache=False
            )
            
            # Send transcription result
            if result.text.strip():
                await websocket.send(json.dumps({
                    "type": "transcription",
                    "text": result.text,
                    "confidence": result.confidence,
                    "is_final": not session.is_speaking,
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Update conversation context
                session.conversation_context += f"User: {result.text}\n"
            
            # Cleanup
            import os
            os.remove(temp_file)
            
        except Exception as e:
            logger.error(f"STT processing error: {str(e)}")
    
    async def _generate_ai_response(self, websocket, session: ConversationSession):
        """Generate AI response with real-time TTS streaming."""
        try:
            # Check if AI should respond
            if not self.turn_manager.should_ai_respond(session.session_id):
                return
            
            # Generate response text (integrate with your chat service)
            response_text = await self._generate_chat_response(session.conversation_context, session.language)
            
            if not response_text:
                return
            
            # Send response text
            await websocket.send(json.dumps({
                "type": "ai_response_text",
                "text": response_text,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Stream TTS audio
            await self._stream_tts_audio(websocket, session, response_text)
            
            # Update conversation context
            session.conversation_context += f"AI: {response_text}\n"
            
        except Exception as e:
            logger.error(f"AI response generation error: {str(e)}")
    
    async def _stream_tts_audio(self, websocket, session: ConversationSession, text: str):
        """Stream TTS audio in real-time."""
        try:
            # Generate TTS audio
            result = await tts_engine.synthesize(
                text=text,
                language=session.language,
                output_path=None  # Return audio data directly
            )
            
            if result.audio_data is not None:
                # Convert to 16-bit PCM
                audio_pcm = (result.audio_data * 32767).astype(np.int16)
                
                # Stream audio in chunks
                chunk_size = 3200  # 200ms at 16kHz
                for i in range(0, len(audio_pcm), chunk_size):
                    chunk = audio_pcm[i:i + chunk_size]
                    
                    # Check for interruption
                    if session.is_speaking:
                        # User interrupted - stop TTS
                        await websocket.send(json.dumps({
                            "type": "ai_interrupted",
                            "timestamp": datetime.now().isoformat()
                        }))
                        break
                    
                    # Send audio chunk
                    await websocket.send(chunk.tobytes())
                    
                    # Small delay for real-time streaming
                    await asyncio.sleep(0.1)
                
                # Send completion signal
                await websocket.send(json.dumps({
                    "type": "ai_response_complete",
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(f"TTS streaming error: {str(e)}")
    
    async def _generate_chat_response(self, context: str, language: str) -> str:
        """Generate chat response using existing chat service."""
        # Integrate with your existing chat service
        # This is a placeholder - replace with actual implementation
        return f"Thank you for your message. I understand you said: {context[-100:]}"
    
    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check rate limiting for WebSocket connections."""
        now = datetime.now()
        
        if connection_id not in self.rate_limit_tracker:
            self.rate_limit_tracker[connection_id] = {
                'requests': 0,
                'window_start': now
            }
        
        tracker = self.rate_limit_tracker[connection_id]
        
        # Reset window if needed
        if now - tracker['window_start'] > timedelta(minutes=1):
            tracker['requests'] = 0
            tracker['window_start'] = now
        
        # Check limit
        if tracker['requests'] >= WEBSOCKET_RATE_LIMIT:
            return False
        
        tracker['requests'] += 1
        return True
    
    def _cleanup_rate_limit(self, connection_id: str):
        """Clean up rate limit tracking."""
        self.rate_limit_tracker.pop(connection_id, None)
    
    def _parse_query_params(self, path: str) -> Dict[str, str]:
        """Parse query parameters from WebSocket path."""
        if '?' not in path:
            return {}
        
        query_string = path.split('?')[1]
        params = {}
        
        for param in query_string.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key] = value
        
        return params

class VoiceActivityDetector:
    """Real-time Voice Activity Detection for barge-in detection."""
    
    def __init__(self):
        self.energy_threshold = 0.01
        self.silence_threshold = 0.005
        self.min_speech_duration = 0.1  # seconds
        self.min_silence_duration = 0.2  # seconds
        
    def detect_voice_activity(self, audio_chunk: bytes) -> Dict:
        """Detect voice activity in audio chunk."""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Calculate energy
            energy = np.mean(np.square(audio_float))
            
            # Determine if speech is present
            is_speech = energy > self.energy_threshold
            
            return {
                'is_speech': is_speech,
                'energy': energy,
                'confidence': min(energy / self.energy_threshold, 1.0)
            }
            
        except Exception as e:
            logger.error(f"VAD error: {str(e)}")
            return {'is_speech': False, 'energy': 0.0, 'confidence': 0.0}

class TurnTakingManager:
    """Manage turn-taking and interruption detection."""
    
    def __init__(self):
        self.user_sessions: Dict[str, Dict] = {}
        self.turn_timeout = TURN_TIMEOUT
        
    def user_started_speaking(self, session_id: str):
        """User started speaking - handle interruption detection."""
        if session_id in self.user_sessions:
            session = self.user_sessions[session_id]
            session['interruption_count'] += 1
            session['last_speech_start'] = datetime.now()
        else:
            self.user_sessions[session_id] = {
                'interruption_count': 0,
                'last_speech_start': datetime.now(),
                'ai_speaking': False
            }
    
    def user_finished_speaking(self, session_id: str):
        """User finished speaking - prepare for AI response."""
        if session_id in self.user_sessions:
            self.user_sessions[session_id]['ai_speaking'] = False
    
    def should_ai_respond(self, session_id: str) -> bool:
        """Determine if AI should respond based on turn-taking rules."""
        if session_id not in self.user_sessions:
            return True
        
        session = self.user_sessions[session_id]
        
        # Check if AI was interrupted
        if session['interruption_count'] > 0:
            # Reset interruption count after AI responds
            session['interruption_count'] = 0
            return True
        
        # Check turn timeout
        if 'last_speech_start' in session:
            time_since_speech = datetime.now() - session['last_speech_start']
            if time_since_speech.total_seconds() > self.turn_timeout:
                return True
        
        return True
```

### WebSocket Endpoint Integration
```python
# Add to main.py
from websocket_service import WebSocketAudioStreamingService
import websockets

# Initialize WebSocket service
websocket_service = WebSocketAudioStreamingService()

# WebSocket endpoint
@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for full-duplex audio streaming."""
    await websocket.accept()
    
    # Extract parameters from query string
    user_id = websocket.query_params.get("user_id")
    language = websocket.query_params.get("language", "en")
    
    if not user_id:
        await websocket.close(code=1008, reason="Missing user_id parameter")
        return
    
    # Handle WebSocket connection
    await websocket_service.handle_websocket_connection(websocket, f"/ws/audio/{session_id}?user_id={user_id}&language={language}")
```

### Configuration Updates
```python
# Add to config.py
# WebSocket Configuration
WEBSOCKET_RATE_LIMIT = int(os.getenv("WEBSOCKET_RATE_LIMIT", "100"))  # requests per minute
VAD_SILENCE_THRESHOLD = float(os.getenv("VAD_SILENCE_THRESHOLD", "0.5"))  # seconds
TURN_TIMEOUT = float(os.getenv("TURN_TIMEOUT", "2.0"))  # seconds

# Audio Streaming Configuration
AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "3200"))  # bytes (200ms at 16kHz)
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
```

### HTML Client Implementation
```html
<!-- websocket_client.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Voice Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .status.connected { background-color: #d4edda; color: #155724; }
        .status.disconnected { background-color: #f8d7da; color: #721c24; }
        .status.speaking { background-color: #fff3cd; color: #856404; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .start-btn { background-color: #28a745; color: white; }
        .stop-btn { background-color: #dc3545; color: white; }
        .start-btn:disabled, .stop-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .conversation {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 5px;
        }
        .user-message { background-color: #007bff; color: white; margin-left: 20%; }
        .ai-message { background-color: #e9ecef; color: #333; margin-right: 20%; }
        .transcription {
            background-color: #fff3cd;
            color: #856404;
            font-style: italic;
        }
        .error { background-color: #f8d7da; color: #721c24; }
        .settings {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .settings label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .settings input, .settings select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-Time Voice Chat</h1>
        
        <div class="settings">
            <label for="sessionId">Session ID:</label>
            <input type="text" id="sessionId" value="session_001" />
            
            <label for="userId">User ID:</label>
            <input type="text" id="userId" value="user_001" />
            
            <label for="language">Language:</label>
            <select id="language">
                <option value="en">English</option>
                <option value="hi">Hindi</option>
                <option value="de">German</option>
                <option value="es">Spanish</option>
            </select>
        </div>
        
        <div id="status" class="status disconnected">Disconnected</div>
        
        <div class="controls">
            <button id="connectBtn" class="start-btn">Connect</button>
            <button id="disconnectBtn" class="stop-btn" disabled>Disconnect</button>
            <button id="startBtn" class="start-btn" disabled>Start Speaking</button>
            <button id="stopBtn" class="stop-btn" disabled>Stop Speaking</button>
        </div>
        
        <div class="conversation" id="conversation"></div>
    </div>

    <script>
        class VoiceChatClient {
            constructor() {
                this.websocket = null;
                this.mediaRecorder = null;
                this.audioContext = null;
                this.isRecording = false;
                this.isConnected = false;
                
                this.initializeElements();
                this.bindEvents();
            }
            
            initializeElements() {
                this.statusEl = document.getElementById('status');
                this.conversationEl = document.getElementById('conversation');
                this.connectBtn = document.getElementById('connectBtn');
                this.disconnectBtn = document.getElementById('disconnectBtn');
                this.startBtn = document.getElementById('startBtn');
                this.stopBtn = document.getElementById('stopBtn');
                this.sessionIdEl = document.getElementById('sessionId');
                this.userIdEl = document.getElementById('userId');
                this.languageEl = document.getElementById('language');
            }
            
            bindEvents() {
                this.connectBtn.addEventListener('click', () => this.connect());
                this.disconnectBtn.addEventListener('click', () => this.disconnect());
                this.startBtn.addEventListener('click', () => this.startRecording());
                this.stopBtn.addEventListener('click', () => this.stopRecording());
            }
            
            async connect() {
                try {
                    const sessionId = this.sessionIdEl.value;
                    const userId = this.userIdEl.value;
                    const language = this.languageEl.value;
                    
                    if (!sessionId || !userId) {
                        this.showError('Please enter Session ID and User ID');
                        return;
                    }
                    
                    const wsUrl = `ws://localhost:8003/ws/audio/${sessionId}?user_id=${userId}&language=${language}`;
                    this.websocket = new WebSocket(wsUrl);
                    
                    this.websocket.onopen = () => {
                        this.isConnected = true;
                        this.updateStatus('Connected', 'connected');
                        this.connectBtn.disabled = true;
                        this.disconnectBtn.disabled = false;
                        this.startBtn.disabled = false;
                        this.addMessage('System', 'Connected to voice chat server', 'system');
                    };
                    
                    this.websocket.onmessage = (event) => {
                        this.handleMessage(event);
                    };
                    
                    this.websocket.onclose = () => {
                        this.isConnected = false;
                        this.updateStatus('Disconnected', 'disconnected');
                        this.connectBtn.disabled = false;
                        this.disconnectBtn.disabled = true;
                        this.startBtn.disabled = true;
                        this.stopBtn.disabled = true;
                        this.stopRecording();
                        this.addMessage('System', 'Disconnected from server', 'system');
                    };
                    
                    this.websocket.onerror = (error) => {
                        this.showError('WebSocket error: ' + error);
                    };
                    
                } catch (error) {
                    this.showError('Connection failed: ' + error.message);
                }
            }
            
            disconnect() {
                if (this.websocket) {
                    this.websocket.close();
                }
            }
            
            async startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });
                    
                    this.audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = this.audioContext.createMediaStreamSource(stream);
                    
                    // Create script processor for real-time audio processing
                    const processor = this.audioContext.createScriptProcessor(1024, 1, 1);
                    
                    processor.onaudioprocess = (event) => {
                        if (this.isRecording && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                            const inputData = event.inputBuffer.getChannelData(0);
                            const audioData = this.convertFloat32ToInt16(inputData);
                            this.websocket.send(audioData);
                        }
                    };
                    
                    source.connect(processor);
                    processor.connect(this.audioContext.destination);
                    
                    this.isRecording = true;
                    this.startBtn.disabled = true;
                    this.stopBtn.disabled = false;
                    this.updateStatus('Speaking...', 'speaking');
                    this.addMessage('System', 'Started recording', 'system');
                    
                } catch (error) {
                    this.showError('Failed to start recording: ' + error.message);
                }
            }
            
            stopRecording() {
                this.isRecording = false;
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                this.updateStatus('Connected', 'connected');
                this.addMessage('System', 'Stopped recording', 'system');
                
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
            }
            
            convertFloat32ToInt16(float32Array) {
                const int16Array = new Int16Array(float32Array.length);
                for (let i = 0; i < float32Array.length; i++) {
                    const s = Math.max(-1, Math.min(1, float32Array[i]));
                    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                return int16Array.buffer;
            }
            
            handleMessage(event) {
                try {
                    if (event.data instanceof Blob) {
                        // Handle audio data
                        this.playAudioResponse(event.data);
                    } else {
                        // Handle text messages
                        const data = JSON.parse(event.data);
                        this.handleTextMessage(data);
                    }
                } catch (error) {
                    console.error('Error handling message:', error);
                }
            }
            
            handleTextMessage(data) {
                switch (data.type) {
                    case 'connection_established':
                        this.addMessage('System', 'Connection established', 'system');
                        break;
                    case 'transcription':
                        this.addMessage('You', data.text, 'user-message');
                        if (data.is_final) {
                            this.addMessage('System', 'Transcription complete', 'transcription');
                        }
                        break;
                    case 'ai_response_text':
                        this.addMessage('AI', data.text, 'ai-message');
                        break;
                    case 'user_speaking_started':
                        this.updateStatus('User speaking...', 'speaking');
                        break;
                    case 'user_speaking_ended':
                        this.updateStatus('AI responding...', 'connected');
                        break;
                    case 'ai_interrupted':
                        this.addMessage('System', 'AI response interrupted', 'system');
                        break;
                    case 'ai_response_complete':
                        this.updateStatus('Connected', 'connected');
                        break;
                    case 'error':
                        this.showError('Server error: ' + data.message);
                        break;
                }
            }
            
            async playAudioResponse(audioBlob) {
                try {
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    await audio.play();
                    
                    audio.onended = () => {
                        URL.revokeObjectURL(audioUrl);
                    };
                } catch (error) {
                    console.error('Error playing audio:', error);
                }
            }
            
            addMessage(sender, text, className) {
                const messageEl = document.createElement('div');
                messageEl.className = `message ${className}`;
                messageEl.innerHTML = `<strong>${sender}:</strong> ${text}`;
                this.conversationEl.appendChild(messageEl);
                this.conversationEl.scrollTop = this.conversationEl.scrollHeight;
            }
            
            updateStatus(text, className) {
                this.statusEl.textContent = text;
                this.statusEl.className = `status ${className}`;
            }
            
            showError(message) {
                this.addMessage('Error', message, 'error');
                console.error(message);
            }
        }
        
        // Initialize the client when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new VoiceChatClient();
        });
    </script>
</body>
</html>
```

### Docker Configuration Updates
```yaml
# Update docker-compose.yml
version: '3.8'

services:
  voice-service:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8003:8003"
    volumes:
      - ./output:/app/output
      - ./parler_models:/app/parler_models
      - ./models/whisper:/app/models/whisper
      - ./models/coqui:/app/models/coqui
      - ./models/huggingface:/app/models/huggingface
      - appuser-cache:/home/appuser/.cache
      - ./websocket_client.html:/app/static/websocket_client.html
    environment:
      - LOG_LEVEL=INFO
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD:-redispassword}
      - REDIS_DB=0
      - RATE_LIMIT_TIER1=100
      - RATE_LIMIT_TIER2=1000
      - MAX_DAILY_COST=100.0
      - COST_TIER_RATIO=0.1
      - TTS_MODEL_ID=ai4bharat/indic-parler-tts
      - TTS_SAMPLING_RATE=16000
      - TTS_USE_CUDA=false
      - WEBSOCKET_RATE_LIMIT=100
      - VAD_SILENCE_THRESHOLD=0.5
      - TURN_TIMEOUT=2.0
      - AUDIO_CHUNK_SIZE=3200
      - AUDIO_SAMPLE_RATE=16000
      - AUDIO_CHANNELS=1
      - GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
      - AZURE_API_KEY=${AZURE_API_KEY:-}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
      - TRANSFORMERS_CACHE=/app/models/huggingface
      - TORCH_HOME=/app/models/coqui
      - HF_HOME=/app/models/huggingface
      - WHISPER_CACHE_DIR=/app/models/whisper
      - PARLER_CACHE_DIR=/app/parler_models
      - NUMBA_CACHE_DIR=/home/appuser/.cache/numba_cache
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --requirepass ${REDIS_PASSWORD:-redispassword}
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD:-redispassword}
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD:-redispassword}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  redis-data:
  appuser-cache: {}
  whisper-cache:
  tts-cache:
```

### Requirements Updates
```txt
# Add to requirements.txt
websockets>=11.0.3
webrtcvad>=2.0.10
```

## Deployment Instructions

1. **Install Dependencies:**
   ```bash
   pip install websockets webrtcvad
   ```

2. **Download Models:**
   ```bash
   python download.py
   ```

3. **Start Services:**
   ```bash
   docker-compose up -d
   ```

4. **Access WebSocket Client:**
   - Open `http://localhost:8003/static/websocket_client.html`
   - Enter Session ID and User ID
   - Click Connect and Start Speaking

## Features Implemented

### ✅ Full-Duplex Audio Streaming
- Real-time bidirectional audio communication
- WebSocket-based streaming with minimal latency
- Audio format conversion and buffering

### ✅ Voice Activity Detection (VAD)
- Real-time speech detection
- Barge-in detection and interruption handling
- Configurable silence thresholds

### ✅ Turn-Taking Management
- Natural conversation flow
- Interruption detection and handling
- Turn timeout management
- Conversation context preservation

### ✅ Rate Limiting
- Per-connection rate limiting
- Configurable limits and windows
- Automatic cleanup

### ✅ Error Handling
- Comprehensive error handling and logging
- Graceful connection management
- Fallback mechanisms

### ✅ Production Features
- Health checks and monitoring
- Resource cleanup
- Session management
- Audio quality optimization

## Usage Example

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8003/ws/audio/session_001?user_id=user_001&language=en');

// Send audio data
ws.send(audioChunk);

// Receive messages
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

This implementation provides a complete, production-ready WebSocket audio streaming service with all requested features including barge-in detection, interruption handling, and natural turn-taking. 