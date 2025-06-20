#!/usr/bin/env python3
"""
Enhanced Voice Service Startup Script
Initializes all components and starts the service with proper error handling.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app
from websocket_service import WebSocketAudioStreamingService
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB,
    LOG_LEVEL, LOG_FILE
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EnhancedVoiceService:
    """Enhanced Voice Service with all components initialized."""
    
    def __init__(self):
        self.app = app
        self.websocket_service = None
        self.is_running = False
    
    async def initialize_components(self):
        """Initialize all service components."""
        logger.info("Initializing Enhanced Voice Service components...")
        
        try:
            # Initialize WebSocket service
            logger.info("Initializing WebSocket service...")
            self.websocket_service = WebSocketAudioStreamingService()
            
            # Test Redis connection
            logger.info("Testing Redis connection...")
            redis_test = await self.websocket_service.rate_limiter.check_rate_limit(
                identifier="startup_test",
                requests_per_minute=1
            )
            logger.info(f"Redis connection: {'OK' if redis_test['allowed'] else 'FAILED'}")
            
            # Test intent detection
            logger.info("Testing intent detection...")
            intent_test = await self.websocket_service.intent_detection.detect_intent(
                text="Hello, how are you?",
                language="en"
            )
            logger.info(f"Intent detection: {'OK' if intent_test['success'] else 'FAILED'}")
            
            # Test guard rails
            logger.info("Testing guard rails...")
            safety_test = await self.websocket_service.guard_rails.check_content_safety(
                text="Hello, this is a test message",
                user_id="startup_test",
                session_id="startup_test"
            )
            logger.info(f"Guard rails: {'OK' if safety_test['is_safe'] else 'FAILED'}")
            
            # Test LLM response generation
            logger.info("Testing LLM response generation...")
            response_test = await self.websocket_service.llm_response.generate_response(
                user_message="Hello",
                conversation_history=[],
                detected_intent={"intent": "greeting", "confidence": 0.9, "success": True},
                rag_context="",
                language="en",
                user_id="startup_test"
            )
            logger.info(f"LLM response generation: {'OK' if response_test else 'FAILED'}")
            
            logger.info("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {str(e)}")
            return False
    
    async def run_health_check(self):
        """Run a comprehensive health check."""
        logger.info("Running health check...")
        
        health_status = {
            "redis": False,
            "intent_detection": False,
            "guard_rails": False,
            "llm_response": False,
            "rag_service": False
        }
        
        try:
            # Check Redis
            redis_result = await self.websocket_service.rate_limiter.check_rate_limit(
                identifier="health_check",
                requests_per_minute=1
            )
            health_status["redis"] = redis_result["allowed"]
            
            # Check intent detection
            intent_result = await self.websocket_service.intent_detection.detect_intent(
                text="Health check",
                language="en"
            )
            health_status["intent_detection"] = intent_result["success"]
            
            # Check guard rails
            safety_result = await self.websocket_service.guard_rails.check_content_safety(
                text="Health check message",
                user_id="health_check",
                session_id="health_check"
            )
            health_status["guard_rails"] = safety_result["is_safe"]
            
            # Check LLM response
            response_result = await self.websocket_service.llm_response.generate_response(
                user_message="Health check",
                conversation_history=[],
                detected_intent={"intent": "health_check", "confidence": 1.0, "success": True},
                rag_context="",
                language="en",
                user_id="health_check"
            )
            health_status["llm_response"] = bool(response_result)
            
            # Check RAG service
            if self.websocket_service.rag_service.is_enabled():
                rag_result = await self.websocket_service.rag_service.retrieve_relevant_context(
                    query="Health check",
                    user_id="health_check"
                )
                health_status["rag_service"] = rag_result["success"]
            else:
                health_status["rag_service"] = True  # Disabled is OK
            
            all_healthy = all(health_status.values())
            logger.info(f"Health check results: {health_status}")
            logger.info(f"Overall health: {'✅ HEALTHY' if all_healthy else '❌ UNHEALTHY'}")
            
            return all_healthy, health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False, health_status
    
    def print_startup_banner(self):
        """Print a nice startup banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Enhanced Voice Service                    ║
║                                                              ║
║  🎤 Real-time STT/TTS with full AI pipeline                 ║
║  🤖 Intent detection, guard rails, RAG, and LLM responses   ║
║  🔄 Full-duplex WebSocket streaming with barge-in support   ║
║  🚀 Production-grade with Redis rate limiting               ║
║                                                              ║
║  Starting service...                                         ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_service_info(self):
        """Print service information."""
        info = f"""
📊 Service Information:
   • Redis: {REDIS_HOST}:{REDIS_PORT}
   • Log Level: {LOG_LEVEL}
   • Log File: {LOG_FILE}
   • WebSocket: ws://localhost:8003/ws/audio/
   • REST API: http://localhost:8003/api/v1/
   • Web Client: http://localhost:8003/static/websocket_client.html

🔧 Available Endpoints:
   • GET  /api/v1/health - Health check
   • POST /api/v1/stt/transcribe - Speech to text
   • POST /api/v1/tts/synthesize - Text to speech
   • GET  /api/v1/rate-limit/status - Rate limit status
   • WS   /ws/audio/{session_id} - WebSocket streaming

📝 Usage Examples:
   • WebSocket: ws://localhost:8003/ws/audio/session_001?user_id=user_001&language=en
   • STT: curl -X POST "http://localhost:8003/api/v1/stt/transcribe" -F "audio_file=@audio.wav"
   • TTS: curl -X POST "http://localhost:8003/api/v1/tts/synthesize" -d '{"text":"Hello","language":"en"}'

🎯 Testing:
   • Run: python test_integration.py
   • Open: http://localhost:8003/static/websocket_client.html
        """
        print(info)

async def main():
    """Main startup function."""
    service = EnhancedVoiceService()
    
    # Print startup banner
    service.print_startup_banner()
    
    # Initialize components
    if not await service.initialize_components():
        logger.error("Failed to initialize components. Exiting.")
        sys.exit(1)
    
    # Run health check
    healthy, health_status = await service.run_health_check()
    if not healthy:
        logger.warning("Some components are unhealthy, but continuing...")
    
    # Print service information
    service.print_service_info()
    
    # Start the service
    logger.info("🚀 Starting Enhanced Voice Service...")
    service.is_running = True
    
    try:
        import uvicorn
        uvicorn.run(
            service.app,
            host="0.0.0.0",
            port=8003,
            log_level=LOG_LEVEL.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
    finally:
        service.is_running = False
        logger.info("Service stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1) 