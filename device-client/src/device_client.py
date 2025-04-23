import os
import json
import logging
import time
import threading
import queue
from pathlib import Path

from .api_client import APIClient
from .sync_manager import SyncManager
from .offline_server_manager import OfflineServerManager
from .language_manager import LanguageManager
from .chat_manager import ChatManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("echonest_device.client")

class DeviceClient:
    """
    Main client for the EchoNest AI device.
    Manages device registration, synchronization, and communication with backend.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the device client.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up paths
        self.base_path = Path(self.config.get("base_path", os.path.expanduser("~/.echonest")))
        self.models_path = Path(self.config.get("models_path", self.base_path / "models"))
        self.content_path = Path(self.config.get("content_path", self.base_path / "content"))
        self.cache_path = Path(self.config.get("cache_path", self.base_path / "cache"))
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.content_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize device state
        self.device_state_path = self.base_path / "device_state.json"
        self.device_state = self._load_device_state()
        
        # Initialize API client
        self.api_client = APIClient(
            api_url=self.config.get("api_url"),
            device_id=self.device_state.get("device_id"),
            device_token=self.device_state.get("device_token")
        )
        
        # Initialize managers
        self.sync_manager = SyncManager(
            device_id=self.device_state.get("device_id"),
            device_token=self.device_state.get("device_token"),
            api_url=self.config.get("api_url"),
            content_path=self.content_path,
            models_path=self.models_path,
            cache_path=self.cache_path
        )
        
        self.offline_server_manager = OfflineServerManager(
            server_url=self.config.get("offline_server_url", "http://localhost:8000"),
            models_path=self.models_path,
            content_path=self.content_path,
            cache_path=self.cache_path
        )
        
        self.language_manager = LanguageManager(
            models_path=self.models_path,
            cache_path=self.cache_path,
            api_client=self.api_client
        )
        
        self.chat_manager = ChatManager(
            models_path=self.models_path,
            content_path=self.content_path,
            cache_path=self.cache_path,
            language_manager=self.language_manager
        )
        
        # Initialize event queue and worker thread
        self.event_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        logger.info("EchoNest AI Device Client initialized")
    
    def _load_config(self, config_path=None):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration
        """
        default_config = {
            "api_url": "https://api.echonest.ai",
            "offline_server_url": "http://localhost:8000",
            "base_path": os.path.expanduser("~/.echonest"),
            "log_level": "INFO"
        }
        
        if not config_path:
            config_path = os.path.expanduser("~/.echonest/config.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.info(f"Configuration file not found at {config_path}, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def _load_device_state(self):
        """
        Load device state from file.
        
        Returns:
            dict: Device state
        """
        default_state = {
            "device_id": None,
            "device_token": None,
            "registered": False,
            "last_sync": None,
            "device_name": f"EchoNest-{int(time.time())}"
        }
        
        try:
            if self.device_state_path.exists():
                with open(self.device_state_path, 'r') as f:
                    state = json.load(f)
                
                logger.info(f"Loaded device state from {self.device_state_path}")
                return state
            else:
                logger.info(f"Device state file not found at {self.device_state_path}, using defaults")
                return default_state
        except Exception as e:
            logger.error(f"Error loading device state: {e}")
            return default_state
    
    def _save_device_state(self):
        """
        Save device state to file.
        """
        try:
            with open(self.device_state_path, 'w') as f:
                json.dump(self.device_state, f, indent=2)
            
            logger.info(f"Saved device state to {self.device_state_path}")
        except Exception as e:
            logger.error(f"Error saving device state: {e}")
    
    def register_device(self, registration_code=None, device_name=None):
        """
        Register the device with the backend.
        
        Args:
            registration_code: Registration code from parent dashboard
            device_name: Optional device name
            
        Returns:
            dict: Registration result
        """
        if self.device_state.get("registered"):
            logger.info("Device already registered")
            return {"success": True, "message": "Device already registered"}
        
        try:
            # Update device name if provided
            if device_name:
                self.device_state["device_name"] = device_name
            
            # Register device with backend
            result = self.api_client.register_device(
                registration_code=registration_code,
                device_name=self.device_state["device_name"]
            )
            
            if result.get("success"):
                # Update device state
                self.device_state["device_id"] = result.get("device_id")
                self.device_state["device_token"] = result.get("device_token")
                self.device_state["registered"] = True
                self.device_state["registration_time"] = time.time()
                
                # Update API client with new credentials
                self.api_client.device_id = self.device_state["device_id"]
                self.api_client.device_token = self.device_state["device_token"]
                
                # Update sync manager with new credentials
                self.sync_manager.device_id = self.device_state["device_id"]
                self.sync_manager.device_token = self.device_state["device_token"]
                
                # Save device state
                self._save_device_state()
                
                logger.info(f"Device registered successfully with ID {self.device_state['device_id']}")
                return {"success": True, "message": "Device registered successfully"}
            else:
                logger.error(f"Device registration failed: {result.get('message')}")
                return result
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            return {"success": False, "message": str(e)}
    
    def start(self):
        """
        Start the device client.
        
        Returns:
            bool: True if started successfully
        """
        if self.running:
            logger.info("Device client already running")
            return True
        
        try:
            # Check if device is registered
            if not self.device_state.get("registered"):
                logger.warning("Device not registered, some features may not work")
            
            # Start offline server
            server_started = self.offline_server_manager.start_server()
            if not server_started:
                logger.warning("Failed to start offline server, some features may not work")
            
            # Start worker thread
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            logger.info("Device client started")
            return True
        except Exception as e:
            logger.error(f"Error starting device client: {e}")
            self.running = False
            return False
    
    def stop(self):
        """
        Stop the device client.
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.running:
            logger.info("Device client already stopped")
            return True
        
        try:
            # Stop worker thread
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
            
            # Stop offline server
            self.offline_server_manager.stop_server()
            
            logger.info("Device client stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping device client: {e}")
            return False
    
    def _worker_loop(self):
        """
        Worker thread loop for processing events.
        """
        while self.running:
            try:
                # Process events from queue
                try:
                    event = self.event_queue.get(timeout=1)
                    self._process_event(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    pass
                
                # Perform periodic tasks
                self._perform_periodic_tasks()
                
                # Sleep to avoid high CPU usage
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    def _process_event(self, event):
        """
        Process an event from the queue.
        
        Args:
            event: Event to process
        """
        event_type = event.get("type")
        
        if event_type == "sync":
            # Synchronize content
            force = event.get("force", False)
            self.sync_manager.sync_content(force=force)
        elif event_type == "chat":
            # Process chat message
            query = event.get("query")
            session_id = event.get("session_id")
            language = event.get("language")
            
            if query:
                self.chat_manager.generate_response(
                    query=query,
                    session_id=session_id,
                    language=language
                )
        elif event_type == "language":
            # Process language-related event
            action = event.get("action")
            
            if action == "detect":
                text = event.get("text")
                if text:
                    self.language_manager.detect_language(text)
            elif action == "translate":
                text = event.get("text")
                source_lang = event.get("source_lang")
                target_lang = event.get("target_lang", "en")
                
                if text:
                    self.language_manager.translate_text(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
            elif action == "speech_to_text":
                audio_path = event.get("audio_path")
                language = event.get("language")
                
                if audio_path:
                    self.language_manager.speech_to_text(
                        audio_path=audio_path,
                        language=language
                    )
            elif action == "text_to_speech":
                text = event.get("text")
                language = event.get("language", "en")
                voice = event.get("voice")
                
                if text:
                    self.language_manager.text_to_speech(
                        text=text,
                        language=language,
                        voice=voice
                    )
    
    def _perform_periodic_tasks(self):
        """
        Perform periodic tasks.
        """
        # Check if it's time to sync
        last_sync = self.device_state.get("last_sync", 0)
        sync_interval = self.config.get("sync_interval", 3600)  # Default: 1 hour
        
        if time.time() - last_sync > sync_interval:
            # Queue sync event
            self.event_queue.put({"type": "sync"})
            
            # Update last sync time
            self.device_state["last_sync"] = time.time()
            self._save_device_state()
    
    def sync_now(self, force=False):
        """
        Trigger immediate synchronization.
        
        Args:
            force: Force full synchronization
            
        Returns:
            bool: True if sync was queued
        """
        try:
            self.event_queue.put({"type": "sync", "force": force})
            logger.info(f"Queued sync event (force={force})")
            return True
        except Exception as e:
            logger.error(f"Error queueing sync event: {e}")
            return False
    
    def process_chat(self, query, session_id=None, language=None):
        """
        Process a chat query.
        
        Args:
            query: User query
            session_id: Optional session ID for context
            language: Optional language code
            
        Returns:
            dict: Response information
        """
        try:
            # Create session if not provided
            if not session_id:
                session = self.chat_manager.create_session(language=language)
                session_id = session["id"]
            
            # Add user message to session
            self.chat_manager.add_message(
                session_id=session_id,
                content=query,
                is_user=True
            )
            
            # Generate response immediately (not queued)
            response = self.chat_manager.generate_response(
                query=query,
                session_id=session_id,
                language=language
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error processing chat query: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def process_voice(self, audio_path, session_id=None, language=None):
        """
        Process a voice query.
        
        Args:
            audio_path: Path to audio file
            session_id: Optional session ID for context
            language: Optional language code
            
        Returns:
            dict: Response information
        """
        try:
            # Convert speech to text
            transcription = self.language_manager.speech_to_text(
                audio_path=audio_path,
                language=language
            )
            
            if not transcription.get("text"):
                return {
                    "success": False,
                    "message": "Failed to transcribe audio"
                }
            
            # Process chat query
            query = transcription["text"]
            detected_language = transcription["language"]
            
            response = self.process_chat(
                query=query,
                session_id=session_id,
                language=detected_language
            )
            
            # Convert response to speech
            if response.get("success") and response.get("response", {}).get("response"):
                response_text = response["response"]["response"]
                
                speech = self.language_manager.text_to_speech(
                    text=response_text,
                    language=detected_language
                )
                
                response["audio_path"] = speech.get("audio_path")
            
            # Add transcription to response
            response["transcription"] = transcription
            
            return response
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def get_device_info(self):
        """
        Get device information.
        
        Returns:
            dict: Device information
        """
        return {
            "device_id": self.device_state.get("device_id"),
            "device_name": self.device_state.get("device_name"),
            "registered": self.device_state.get("registered", False),
            "registration_time": self.device_state.get("registration_time"),
            "last_sync": self.device_state.get("last_sync"),
            "sync_status": self.sync_manager.get_sync_status(),
            "offline_server": self.offline_server_manager.get_server_status(),
            "content_items_count": len(self.sync_manager.get_content_items()),
            "models_count": len(self.sync_manager.get_models()),
            "supported_languages": list(self.language_manager.supported_languages.keys())
        }
    
    def get_chat_sessions(self):
        """
        Get all chat sessions.
        
        Returns:
            list: Chat sessions
        """
        return self.chat_manager.get_sessions()
    
    def get_chat_messages(self, session_id, limit=None):
        """
        Get messages from a chat session.
        
        Args:
            session_id: Session ID
            limit: Optional maximum number of messages to return
            
        Returns:
            list: Chat messages
        """
        return self.chat_manager.get_messages(session_id, limit=limit)
