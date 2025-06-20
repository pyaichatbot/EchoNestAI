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
    
    async def handle_websocket_connection(self, websocket, session_id: str, user_id: str, language: str):
        """Handle incoming WebSocket connections with full-duplex audio streaming."""
        connection_id = f"{user_id}_{session_id}"
        
        try:
            # This check is now more direct
            if not session_id or not user_id:
                await websocket.close(1008, "Missing required parameters")
                return
            
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
            await self._safe_send_json(websocket, {
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Handle incoming messages using the correct loop for Starlette WebSockets
            while True:
                try:
                    message = await websocket.receive()
                    
                    # Check if this is a disconnect message
                    if message.get("type") == "websocket.disconnect":
                        logger.info(f"WebSocket disconnect received for {connection_id}")
                        break
                    
                    # Extract the payload ('text' or 'bytes') from the message dictionary
                    content = message.get('text') or message.get('bytes')
                    if content:
                        await self._handle_incoming_message(websocket, session, content)
                        
                except RuntimeError as e:
                    if "disconnect message has been received" in str(e):
                        logger.info(f"WebSocket disconnected for {connection_id}")
                        break
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error in WebSocket message loop: {str(e)}")
                    break
                
        except Exception as e:
            # Handle graceful disconnects without logging as an error
            if type(e).__name__ == 'WebSocketDisconnect' or isinstance(e, websockets.exceptions.ConnectionClosed):
                logger.info(f"WebSocket connection closed: {connection_id}")
            else:
                logger.error(f"WebSocket error for {connection_id}: {str(e)}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    # Could not send error to client, probably already disconnected
                    pass
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
            await websocket.send_json({
                "type": "error",
                "message": f"Message processing error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _process_audio_chunk(self, websocket, session: ConversationSession, audio_data: bytes):
        """Process incoming audio chunk with real-time STT and VAD."""
        try:
            # Convert audio data to numpy array for level checking
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check audio level - skip if too quiet (likely silence)
            audio_level = np.abs(audio_array).mean()
            if audio_level < 100:  # Threshold for silence detection
                return
            
            # Check for voice activity
            vad_result = self.vad_processor.detect_voice_activity(audio_data)
            
            if vad_result['is_speech']:
                if not session.is_speaking:
                    # User started speaking
                    session.is_speaking = True
                    session.turn_start_time = datetime.now()
                    
                    # Notify turn manager
                    self.turn_manager.user_started_speaking(session.session_id)
                    
                    await self._safe_send_json(websocket, {
                        "type": "user_speaking_started",
                        "timestamp": datetime.now().isoformat()
                    })
                
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
                        
                        await self._safe_send_json(websocket, {
                            "type": "user_speaking_ended",
                            "timestamp": datetime.now().isoformat()
                        })
                        
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
            
            # Enhanced audio level checking
            audio_level = np.abs(audio_array).mean()
            audio_max = np.abs(audio_array).max()
            
            # Skip if audio is too quiet or too short
            if audio_level < 200 or audio_max < 500:  # Increased thresholds
                logger.debug(f"Audio too quiet - level: {audio_level:.2f}, max: {audio_max:.2f}")
                return
            
            # Check for minimum audio duration (at least 0.5 seconds of speech)
            if len(audio_array) < 8000:  # 0.5 seconds at 16kHz
                logger.debug(f"Audio too short - length: {len(audio_array)} samples")
                return
            
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
                
                # Only send transcription if it's meaningful
                if result.text.strip() and len(result.text.strip()) > 2:
                    # Check if the transcription makes sense (not just noise)
                    if not self._is_noise_transcription(result.text):
                        await self._safe_send_json(websocket, {
                            "type": "transcription",
                            "text": result.text,
                            "confidence": result.confidence,
                            "is_final": not session.is_speaking,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update conversation context
                        session.conversation_context += f"User: {result.text}\n"
                        
                        # Add to conversation history
                        session.conversation_history.append({
                            "role": "user",
                            "content": result.text,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        logger.info(f"STT result: '{result.text}' (confidence: {result.confidence:.2f})")
                    else:
                        logger.debug(f"Filtered out noise transcription: '{result.text}'")
                else:
                    logger.debug(f"Empty or too short STT result: '{result.text}'")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            raise
    
    def _is_noise_transcription(self, text: str) -> bool:
        """Check if transcription is likely noise or meaningless."""
        text_lower = text.lower().strip()
        
        # Common noise patterns
        noise_patterns = [
            r'^\s*[aeiou]+\s*$',  # Just vowels
            r'^\s*[bcdfghjklmnpqrstvwxyz]+\s*$',  # Just consonants
            r'^\s*[^\w\s]+\s*$',  # Just punctuation/symbols
            r'^\s*[0-9]+\s*$',  # Just numbers
            r'^\s*[a-z]{1,2}\s*$',  # Very short words
        ]
        
        import re
        for pattern in noise_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check for repetitive characters
        if len(set(text_lower.replace(' ', ''))) < 3:
            return True
        
        return False
    
    async def _generate_ai_response(self, websocket, session: ConversationSession):
        """Generate AI response with full pipeline: intent detection, guard rails, RAG, and LLM generation."""
        try:
            logger.info(f"Starting AI response generation for session {session.session_id}")
            
            # Check if AI should respond
            if not self.turn_manager.should_ai_respond(session.session_id):
                logger.info(f"AI response skipped - turn manager says no for session {session.session_id}")
                return
            
            # Get the latest user message
            if not session.conversation_history:
                logger.info(f"No conversation history for session {session.session_id}")
                return
            
            latest_user_message = session.conversation_history[-1]["content"]
            logger.info(f"Processing user message: '{latest_user_message[:50]}...' for session {session.session_id}")
            
            # Step 1: Intent Detection
            logger.info(f"Starting intent detection for session {session.session_id}")
            try:
                intent_result = await self.intent_detection.detect_intent(
                    text=latest_user_message
                )
                logger.info(f"Intent detection completed: {intent_result.intent_type.value} for session {session.session_id}")
            except Exception as e:
                logger.error(f"Intent detection failed: {e} for session {session.session_id}")
                # Create a fallback intent
                from commoncode.llm.intent_detector import Intent, IntentType
                intent_result = Intent(
                    intent_type=IntentType.STATEMENT,
                    confidence=0.5,
                    entities={},
                    slots={},
                    raw_text=latest_user_message,
                    processed_text=latest_user_message,
                    language=session.language,
                    provider="fallback",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
            
            session.detected_intent = intent_result
            
            await self._safe_send_json(websocket, {
                "type": "intent_detected",
                "intent": {
                    "intent_type": intent_result.intent_type.value,
                    "confidence": intent_result.confidence,
                    "entities": intent_result.entities,
                    "slots": intent_result.slots,
                    "language": intent_result.language,
                    "provider": intent_result.provider
                },
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 2: Guard Rails - Content Safety Check
            logger.info(f"Starting content safety check for session {session.session_id}")
            try:
                safety_result = await self.guard_rails.check_safety(
                    content=latest_user_message,
                    content_type=ContentType.TEXT
                )
                logger.info(f"Content safety check completed: {safety_result.is_safe} for session {session.session_id}")
            except Exception as e:
                logger.error(f"Content safety check failed: {e} for session {session.session_id}")
                # Create a fallback safety result (assume safe)
                from commoncode.guard_rails.guard_rails import SafetyResult, SafetyLevel
                safety_result = SafetyResult(
                    is_safe=True,
                    safety_level=SafetyLevel.SAFE,
                    risk_score=0.0,
                    flagged_categories=[],
                    flagged_content=[],
                    recommendations=[],
                    content_type=ContentType.TEXT,
                    provider="fallback",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
            
            if not safety_result.is_safe:
                # Content flagged as unsafe
                logger.warning(f"Content safety violation for session {session.session_id}")
                await self._safe_send_json(websocket, {
                    "type": "content_safety_violation",
                    "reason": safety_result.recommendations[0] if safety_result.recommendations else "Content flagged as unsafe",
                    "severity": safety_result.safety_level.value,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send a safe response
                safe_response = "I'm sorry, but I cannot respond to that type of content. How can I help you with something else?"
                await self._stream_tts_audio(websocket, session, safe_response)
                return
            
            # Step 3: RAG - Retrieve relevant context (if enabled)
            logger.info(f"Starting RAG context retrieval for session {session.session_id}")
            rag_context = None
            if hasattr(self, 'rag_service') and self.rag_service and hasattr(self.rag_service, 'retrieve_relevant_context'):
                try:
                    rag_result = await self.rag_service.retrieve_relevant_context(
                        query=latest_user_message,
                        user_id=session.user_id
                    )
                    if rag_result and rag_result.get('success') and rag_result.get('context'):
                        rag_context = rag_result['context']
                        logger.info(f"RAG context retrieved: {len(rag_context)} characters for session {session.session_id}")
                except Exception as e:
                    logger.warning(f"RAG context retrieval failed: {e} for session {session.session_id}")
            else:
                logger.info(f"RAG service not available for session {session.session_id}")
            
            # Step 4: LLM Response Generation
            logger.info(f"Starting LLM response generation for session {session.session_id}")
            
            # Prepare prompt with context
            prompt = f"User: {latest_user_message}\n\n"
            if rag_context:
                prompt += f"Relevant context: {rag_context}\n\n"
            prompt += "Assistant:"
            
            # Generate response
            try:
                response_result = await self.llm_response.generate_response(
                    prompt=prompt,
                    context={
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "language": session.language,
                        "intent": intent_result.intent_type.value,
                        "conversation_history": session.conversation_history[-5:]  # Last 5 messages
                    }
                )
                logger.info(f"LLM response generated: '{response_result.response_text[:50]}...' for session {session.session_id}")
            except Exception as e:
                logger.error(f"LLM response generation failed: {e} for session {session.session_id}")
                # Create a fallback response
                from commoncode.llm.response_generator import GenerationResult, ResponseType
                response_result = GenerationResult(
                    response_text="I understand what you're saying. Please continue with your question or request.",
                    response_type=ResponseType.CONVERSATIONAL,
                    confidence=0.5,
                    tokens_used=0,
                    cost=0.0,
                    provider="fallback",
                    model_used="fallback",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
            
            # Add AI response to conversation history
            session.conversation_history.append({
                "role": "assistant",
                "content": response_result.response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send response text
            await self._safe_send_json(websocket, {
                "type": "ai_response_text",
                "text": response_result.response_text,
                "intent": intent_result.intent_type.value,
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 5: TTS Audio Generation and Streaming
            logger.info(f"Starting TTS audio generation for session {session.session_id}")
            await self._stream_tts_audio(websocket, session, response_result.response_text)
            logger.info(f"AI response generation completed for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"AI response generation error: {str(e)}", exc_info=True)
            
            # Send error response
            error_response = "I'm sorry, I encountered an error while processing your request. Please try again."
            await self._stream_tts_audio(websocket, session, error_response)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better TTS streaming."""
        import re
        # Simple sentence splitting - can be improved later
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _stream_tts_audio(self, websocket, session: ConversationSession, text: str):
        """Stream TTS audio in real-time chunks."""
        try:
            logger.info(f"Starting TTS audio streaming for text: '{text[:50]}...' for session {session.session_id}")
            
            # Generate TTS audio for the entire text (simpler approach)
            result = await self.tts_engine.synthesize(
                text=text,
                language=session.language,
                output_path=None,  # Return audio data directly
                use_cache=False  # Disable caching for real-time
            )
            
            logger.info(f"TTS audio generated, length: {len(result.audio_data) if result.audio_data else 0} bytes for session {session.session_id}")
            
            if result.audio_data:
                # Convert to numpy array for chunking
                audio_array = np.frombuffer(result.audio_data, dtype=np.int16)
                
                # Stream audio in chunks
                chunk_size = 4096  # Adjust based on your needs
                for j in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[j:j + chunk_size]
                    
                    # Send audio chunk
                    await self._safe_send_bytes(websocket, chunk.tobytes())
                    
                    # Small delay for real-time streaming
                    await asyncio.sleep(0.1)
                
                logger.info(f"Audio streamed successfully for session {session.session_id}")
            else:
                logger.warning(f"No audio data generated for session {session.session_id}")
            
            # Send completion message
            await self._safe_send_json(websocket, {
                "type": "ai_response_complete",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"TTS audio streaming completed for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"TTS streaming error: {str(e)}", exc_info=True)
            raise
    
    async def _handle_text_message(self, websocket, session: ConversationSession, data: Dict):
        """Handle text messages (commands, metadata)."""
        try:
            message_type = data.get('type')
            
            if message_type == 'ping':
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
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

    async def _safe_send_json(self, websocket, data: dict):
        """Safely send JSON message to WebSocket, handling disconnection errors."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                logger.info(f"WebSocket disconnected while sending message: {e}")
                raise
            else:
                logger.error(f"Error sending JSON message: {e}")
                raise
    
    async def _safe_send_bytes(self, websocket, data: bytes):
        """Safely send binary data to WebSocket, handling disconnection errors."""
        try:
            await websocket.send(data)
        except Exception as e:
            if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                logger.info(f"WebSocket disconnected while sending audio: {e}")
                raise
            else:
                logger.error(f"Error sending audio data: {e}")
                raise

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