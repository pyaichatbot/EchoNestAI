import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Dict, Set, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import queue
import soundfile as sf
import io
import base64
import os
import tempfile
import sys
import redis

from config import (
    WEBSOCKET_RATE_LIMIT, VAD_SILENCE_THRESHOLD, TURN_TIMEOUT,
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB,
    LLM_OPENAI_MODEL, LLM_ANTHROPIC_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    INTENT_DETECTION_MODEL, INTENT_DETECTION_TEMPERATURE, INTENT_DETECTION_MAX_TOKENS,
    INTENT_DETECTION_CONFIDENCE_THRESHOLD,
    GUARD_RAILS_SAFETY_THRESHOLD, GUARD_RAILS_ENABLE_CONTENT_FILTERING,
    GUARD_RAILS_ENABLE_TOXICITY_DETECTION, GUARD_RAILS_ENABLE_BIAS_DETECTION,
    GUARD_RAILS_ENABLE_PII_DETECTION, GUARD_RAILS_ENABLE_HARMFUL_CONTENT_DETECTION,
    GUARD_RAILS_MAX_CONTENT_LENGTH, GUARD_RAILS_ENABLE_LLM_MODERATION,
    GUARD_RAILS_FALLBACK_TO_RULES, BLOCKED_KEYWORDS,
    RAG_ENABLED, RAG_VECTOR_DB_URL, RAG_EMBEDDING_MODEL, RAG_MAX_RESULTS,
    RAG_SIMILARITY_THRESHOLD
)

# Import commoncode components
sys.path.append(os.path.join(os.path.dirname(__file__), 'commoncode'))

from commoncode.rate_limiting import RedisRateLimiter
from commoncode.llm import IntentDetectionEngine, ResponseGenerationEngine
from commoncode.llm.intent_detector import IntentConfig, RuleBasedIntentDetector, OpenAIIntentDetector, AnthropicIntentDetector
from commoncode.guard_rails import GuardRailsEngine, GuardRailsConfig, ContentType
from commoncode.rag import PlaceholderRAGService
from commoncode.rate_limiting.redis_rate_limiter import RateLimitConfig
from commoncode.llm.response_generator import GenerationConfig, ResponseType

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
    detected_intent: Optional[Dict] = None
    conversation_history: List[Dict] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

class WebSocketAudioStreamingService:
    def __init__(self, stt_engine=None, tts_engine=None, redis_client=None):
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.sessions: Dict[str, ConversationSession] = {}
        self.vad_processor = VoiceActivityDetector()
        self.turn_manager = TurnTakingManager()
        
        # Initialize commoncode components
        # Use provided Redis client or create a fallback
        if redis_client is None:
            logger.warning("No Redis client provided, creating fallback in-memory cache")
            # Create a simple in-memory fallback
            class InMemoryCache:
                def __init__(self):
                    self.cache = {}
                    self.expiry = {}
                
                def ping(self):
                    return True
                    
                def get(self, key):
                    if key in self.cache and (key not in self.expiry or self.expiry[key] > datetime.now()):
                        return self.cache[key]
                    return None
                    
                def set(self, key, value):
                    self.cache[key] = value
                    return True
                    
                def setex(self, key, ttl, value):
                    self.cache[key] = value
                    self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
                    return True
                    
                def incr(self, key):
                    if key not in self.cache:
                        self.cache[key] = 1
                    else:
                        self.cache[key] = int(self.cache[key]) + 1
                    return self.cache[key]
                    
                def expire(self, key, ttl):
                    if key in self.cache:
                        self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
                        return True
                    return False
                    
                def delete(self, key):
                    if key in self.cache:
                        del self.cache[key]
                        if key in self.expiry:
                            del self.expiry[key]
                        return True
                    return False
            
            redis_client = InMemoryCache()
        
        # Create rate limit config
        rate_limit_config = RateLimitConfig(
            requests_per_minute=WEBSOCKET_RATE_LIMIT,
            requests_per_hour=WEBSOCKET_RATE_LIMIT * 60,
            requests_per_day=WEBSOCKET_RATE_LIMIT * 1440,
            burst_limit=10,
            window_size_seconds=60,
            cost_based_limits=False
        )
        
        self.rate_limiter = RedisRateLimiter(
            redis_client=redis_client,
            config=rate_limit_config,
            prefix="websocket_rate_limit"
        )
        
        # Initialize intent detection with multiple providers
        intent_config = IntentConfig(
            model_name=INTENT_DETECTION_MODEL,
            temperature=INTENT_DETECTION_TEMPERATURE,
            max_tokens=INTENT_DETECTION_MAX_TOKENS,
            confidence_threshold=INTENT_DETECTION_CONFIDENCE_THRESHOLD
        )
        
        # Create detectors list
        detectors = []
        
        # Add OpenAI detector if API key is available
        if os.getenv("OPENAI_API_KEY"):
            detectors.append(OpenAIIntentDetector(
                api_key=os.getenv("OPENAI_API_KEY"),
                config=intent_config
            ))
        
        # Add Anthropic detector if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            detectors.append(AnthropicIntentDetector(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                config=intent_config
            ))
        
        # Always add rule-based detector as fallback
        detectors.append(RuleBasedIntentDetector(config=intent_config))
        
        self.intent_detection = IntentDetectionEngine(
            detectors=detectors,
            config=intent_config
        )
        
        # Create guard rails configuration
        guard_rails_config = GuardRailsConfig(
            safety_threshold=GUARD_RAILS_SAFETY_THRESHOLD,
            enable_content_filtering=GUARD_RAILS_ENABLE_CONTENT_FILTERING,
            enable_toxicity_detection=GUARD_RAILS_ENABLE_TOXICITY_DETECTION,
            enable_bias_detection=GUARD_RAILS_ENABLE_BIAS_DETECTION,
            enable_pii_detection=GUARD_RAILS_ENABLE_PII_DETECTION,
            enable_harmful_content_detection=GUARD_RAILS_ENABLE_HARMFUL_CONTENT_DETECTION,
            max_content_length=GUARD_RAILS_MAX_CONTENT_LENGTH,
            enable_llm_moderation=GUARD_RAILS_ENABLE_LLM_MODERATION,
            fallback_to_rules=GUARD_RAILS_FALLBACK_TO_RULES,
            blocked_keywords=BLOCKED_KEYWORDS
        )
        
        # Create guard rails instances
        guard_rails_instances = []
        
        # Add OpenAI guard rails if API key is available
        if os.getenv("OPENAI_API_KEY"):
            from commoncode.guard_rails.guard_rails import OpenAIModerationGuardRails
            guard_rails_instances.append(OpenAIModerationGuardRails(
                api_key=os.getenv("OPENAI_API_KEY"),
                config=guard_rails_config
            ))
        
        # Add Anthropic guard rails if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            from commoncode.guard_rails.guard_rails import AnthropicGuardRails
            guard_rails_instances.append(AnthropicGuardRails(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                config=guard_rails_config
            ))
        
        # Always add rule-based guard rails as fallback
        from commoncode.guard_rails.guard_rails import RuleBasedGuardRails
        guard_rails_instances.append(RuleBasedGuardRails(config=guard_rails_config))
        
        self.guard_rails = GuardRailsEngine(
            guard_rails=guard_rails_instances,
            config=guard_rails_config
        )
        
        self.rag_service = PlaceholderRAGService()
        
        # Create response generation configuration
        generation_config = GenerationConfig(
            model_name=LLM_OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            response_type=ResponseType.CONVERSATIONAL
        )
        
        # Create response generators list
        generators = []
        
        # Add OpenAI generator if API key is available
        if os.getenv("OPENAI_API_KEY"):
            from commoncode.llm.response_generator import OpenAIResponseGenerator
            generators.append(OpenAIResponseGenerator(
                api_key=os.getenv("OPENAI_API_KEY"),
                default_config=generation_config
            ))
        
        # Add Anthropic generator if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            from commoncode.llm.response_generator import AnthropicResponseGenerator
            generators.append(AnthropicResponseGenerator(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                default_config=generation_config
            ))
        
        # Always add local generator as fallback
        from commoncode.llm.response_generator import LocalResponseGenerator
        generators.append(LocalResponseGenerator(default_config=generation_config))
        
        self.llm_response = ResponseGenerationEngine(
            generators=generators,
            default_config=generation_config
        )
        
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
            
            # Redis-based rate limiting check
            rate_limit_result = self.rate_limiter.is_allowed(
                key=connection_id,
                cost=1.0
            )
            
            if not rate_limit_result.allowed:
                await websocket.close(1008, f"Rate limit exceeded. Try again in {rate_limit_result.retry_after} seconds")
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
    
    async def _handle_incoming_message(self, websocket, session: ConversationSession, message):
        """Handle incoming WebSocket messages."""
        try:
            if isinstance(message, bytes):
                # Handle binary audio data
                await self._process_audio_chunk(websocket, session, message)
            else:
                # Handle text messages
                data = json.loads(message)
                await self._handle_text_message(websocket, session, data)
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Message processing error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def _process_audio_chunk(self, websocket, session: ConversationSession, audio_data: bytes):
        """Process incoming audio chunk with real-time STT and VAD."""
        try:
            # Check for voice activity
            vad_result = self.vad_processor.detect_voice_activity(audio_data)
            
            if vad_result['is_speech']:
                if not session.is_speaking:
                    # User started speaking
                    session.is_speaking = True
                    session.turn_start_time = datetime.now()
                    
                    # Notify turn manager
                    self.turn_manager.user_started_speaking(session.session_id)
                    
                    await websocket.send(json.dumps({
                        "type": "user_speaking_started",
                        "timestamp": datetime.now().isoformat()
                    }))
                
                session.last_activity = datetime.now()
                
                # Process audio for transcription
                await self._perform_stt(websocket, session, audio_data)
            else:
                # Check for end of speech
                if session.is_speaking:
                    silence_duration = (datetime.now() - session.last_activity).total_seconds()
                    if silence_duration > VAD_SILENCE_THRESHOLD:
                        session.is_speaking = False
                        
                        # Notify turn manager
                        self.turn_manager.user_finished_speaking(session.session_id)
                        
                        await websocket.send(json.dumps({
                            "type": "user_speaking_ended",
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                        # Start AI response
                        await self._generate_ai_response(websocket, session)
                        
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            raise
    
    async def _perform_stt(self, websocket, session: ConversationSession, audio_data: bytes):
        """Perform real-time STT on audio chunk."""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save to temporary file for STT processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_array, 16000)
                temp_path = temp_file.name
            
            try:
                # Perform STT
                result = self.stt_engine.transcribe(
                    audio_path=temp_path,
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
                    
                    # Add to conversation history
                    session.conversation_history.append({
                        "role": "user",
                        "content": result.text,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            raise
    
    async def _generate_ai_response(self, websocket, session: ConversationSession):
        """Generate AI response with full pipeline: intent detection, guard rails, RAG, and LLM generation."""
        try:
            # Check if AI should respond
            if not self.turn_manager.should_ai_respond(session.session_id):
                return
            
            # Get the latest user message
            if not session.conversation_history:
                return
            
            latest_user_message = session.conversation_history[-1]["content"]
            
            # Step 1: Intent Detection
            intent_result = await self.intent_detection.detect_intent(
                text=latest_user_message,
                language=session.language
            )
            
            session.detected_intent = intent_result
            
            await websocket.send(json.dumps({
                "type": "intent_detected",
                "intent": intent_result,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Step 2: Guard Rails - Content Safety Check
            safety_result = await self.guard_rails.check_safety(
                content=latest_user_message,
                content_type=ContentType.TEXT
            )
            
            if not safety_result['is_safe']:
                # Content flagged as unsafe
                await websocket.send(json.dumps({
                    "type": "content_safety_violation",
                    "reason": safety_result['reason'],
                    "severity": safety_result['severity'],
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Send a safe response
                safe_response = "I'm sorry, but I cannot respond to that type of content. How can I help you with something else?"
                await self._stream_tts_audio(websocket, session, safe_response)
                return
            
            # Step 3: RAG - Retrieve relevant context (if enabled)
            rag_context = ""
            if self.rag_service.is_enabled():
                rag_result = await self.rag_service.retrieve_relevant_context(
                    query=latest_user_message,
                    user_id=session.user_id
                )
                if rag_result['success'] and rag_result['context']:
                    rag_context = rag_result['context']
            
            # Step 4: Generate LLM Response
            # Create a comprehensive prompt with context
            prompt = f"User: {latest_user_message}"
            
            # Create context dictionary with all relevant information
            context = {
                "conversation_history": session.conversation_history,
                "detected_intent": intent_result,
                "rag_context": rag_context,
                "language": session.language,
                "user_id": session.user_id
            }
            
            result = await self.llm_response.generate_response(
                prompt=prompt,
                context=context
            )
            
            response_text = result.response_text if result.response_text else "I'm sorry, I couldn't generate a response at the moment. Could you please try again?"
            
            # Send response text
            await websocket.send(json.dumps({
                "type": "ai_response_text",
                "text": response_text,
                "intent": intent_result,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Add AI response to conversation history
            session.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Stream TTS audio
            await self._stream_tts_audio(websocket, session, response_text)
            
            # Update conversation context
            session.conversation_context += f"AI: {response_text}\n"
            
        except Exception as e:
            logger.error(f"AI response generation error: {str(e)}")
            # Send fallback response
            fallback_response = "I'm sorry, I encountered an error while processing your request. Please try again."
            await self._stream_tts_audio(websocket, session, fallback_response)
    
    async def _stream_tts_audio(self, websocket, session: ConversationSession, text: str):
        """Stream TTS audio in real-time."""
        try:
            # Generate TTS audio
            result = await self.tts_engine.synthesize(
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
    
    async def _handle_text_message(self, websocket, session: ConversationSession, data: Dict):
        """Handle text messages (commands, metadata)."""
        try:
            message_type = data.get('type')
            
            if message_type == 'ping':
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif message_type == 'command':
                command = data.get('command')
                if command == 'stop_ai':
                    # Stop AI response if currently speaking
                    pass
                elif command == 'clear_context':
                    session.conversation_context = ""
                    session.conversation_history = []
                    session.detected_intent = None
                    
        except Exception as e:
            logger.error(f"Text message handling error: {str(e)}")
    
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