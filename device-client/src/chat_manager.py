import os
import json
import logging
import time
import threading
import queue
from pathlib import Path

logger = logging.getLogger("echonest_device.chat_manager")

class ChatManager:
    """
    Manages chat functionality for the EchoNest AI device.
    Handles conversations, RAG, and offline chat capabilities.
    """
    
    def __init__(self, models_path, content_path, cache_path, language_manager=None):
        """
        Initialize the chat manager.
        
        Args:
            models_path: Path to models directory
            content_path: Path to content directory
            cache_path: Path to cache directory
            language_manager: Optional language manager for multi-language support
        """
        self.models_path = Path(models_path)
        self.content_path = Path(content_path)
        self.cache_path = Path(cache_path)
        self.language_manager = language_manager
        
        # Create chat-specific directories
        self.chat_models_path = self.models_path / "chat"
        self.chat_models_path.mkdir(parents=True, exist_ok=True)
        
        self.chat_cache_path = self.cache_path / "chat"
        self.chat_cache_path.mkdir(parents=True, exist_ok=True)
        
        self.chat_history_path = self.cache_path / "chat_history"
        self.chat_history_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model cache
        self.model_cache = {}
        self.model_cache_lock = threading.Lock()
        
        # Initialize embedding model for RAG
        self._init_embedding_model()
        
        # Initialize chat sessions
        self.sessions = self._load_sessions()
        self.sessions_lock = threading.Lock()
    
    def _init_embedding_model(self):
        """
        Initialize the embedding model for RAG.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Determine model to use
            model_name = "all-MiniLM-L6-v2"  # Default model
            model_path = self.chat_models_path / "embedding" / model_name
            
            # Check if model exists locally
            if not model_path.exists():
                logger.info(f"Embedding model {model_name} not found locally, using remote model")
                model_path = model_name
            
            # Load model
            self.embedding_model = SentenceTransformer(model_path)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None
    
    def _load_sessions(self):
        """
        Load chat sessions from disk.
        
        Returns:
            dict: Chat sessions
        """
        sessions = {}
        
        try:
            sessions_file = self.chat_cache_path / "sessions.json"
            
            if sessions_file.exists():
                with open(sessions_file, 'r') as f:
                    sessions = json.load(f)
            
            logger.info(f"Loaded {len(sessions)} chat sessions")
        except Exception as e:
            logger.error(f"Error loading chat sessions: {e}")
        
        return sessions
    
    def _save_sessions(self):
        """
        Save chat sessions to disk.
        """
        try:
            sessions_file = self.chat_cache_path / "sessions.json"
            
            with open(sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
            
            logger.info(f"Saved {len(self.sessions)} chat sessions")
        except Exception as e:
            logger.error(f"Error saving chat sessions: {e}")
    
    def create_session(self, name=None, language=None):
        """
        Create a new chat session.
        
        Args:
            name: Optional session name
            language: Optional language code
            
        Returns:
            dict: Session information
        """
        with self.sessions_lock:
            session_id = f"session_{int(time.time())}"
            
            self.sessions[session_id] = {
                "id": session_id,
                "name": name or f"Chat {len(self.sessions) + 1}",
                "language": language or "en",
                "created_at": time.time(),
                "updated_at": time.time(),
                "messages": []
            }
            
            self._save_sessions()
            
            return self.sessions[session_id]
    
    def get_session(self, session_id):
        """
        Get a chat session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            dict: Session information, or None if not found
        """
        with self.sessions_lock:
            return self.sessions.get(session_id)
    
    def get_sessions(self):
        """
        Get all chat sessions.
        
        Returns:
            list: Chat sessions
        """
        with self.sessions_lock:
            return list(self.sessions.values())
    
    def delete_session(self, session_id):
        """
        Delete a chat session.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if session was deleted
        """
        with self.sessions_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self._save_sessions()
                return True
            return False
    
    def add_message(self, session_id, content, is_user=True, metadata=None):
        """
        Add a message to a chat session.
        
        Args:
            session_id: Session ID
            content: Message content
            is_user: Whether the message is from the user
            metadata: Optional message metadata
            
        Returns:
            dict: Added message, or None if session not found
        """
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            
            if not session:
                logger.error(f"Session {session_id} not found")
                return None
            
            message = {
                "id": f"msg_{int(time.time())}_{len(session['messages'])}",
                "content": content,
                "is_user": is_user,
                "timestamp": time.time()
            }
            
            if metadata:
                message["metadata"] = metadata
            
            session["messages"].append(message)
            session["updated_at"] = time.time()
            
            self._save_sessions()
            
            return message
    
    def get_messages(self, session_id, limit=None):
        """
        Get messages from a chat session.
        
        Args:
            session_id: Session ID
            limit: Optional maximum number of messages to return
            
        Returns:
            list: Chat messages, or empty list if session not found
        """
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            
            if not session:
                logger.error(f"Session {session_id} not found")
                return []
            
            messages = session["messages"]
            
            if limit:
                messages = messages[-limit:]
            
            return messages
    
    def search_content(self, query, language=None, top_k=3):
        """
        Search for relevant content using RAG.
        
        Args:
            query: Search query
            language: Optional language code
            top_k: Maximum number of results to return
            
        Returns:
            list: Search results
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return []
        
        try:
            # Translate query if language manager is available and language is specified
            if self.language_manager and language and language != "en":
                translation = self.language_manager.translate_text(query, source_lang=language, target_lang="en")
                search_query = translation["translated_text"]
            else:
                search_query = query
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(search_query)
            
            # Load content index
            index_path = self.chat_cache_path / "content_index.json"
            
            if not index_path.exists():
                logger.warning("Content index not found, creating empty index")
                content_index = []
            else:
                with open(index_path, 'r') as f:
                    content_index = json.load(f)
            
            # If index is empty, return empty results
            if not content_index:
                return []
            
            # Calculate similarity scores
            import numpy as np
            
            results = []
            
            for item in content_index:
                # Skip items without embeddings
                if "embedding" not in item:
                    continue
                
                # Convert embedding from list to numpy array
                item_embedding = np.array(item["embedding"])
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, item_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
                )
                
                results.append({
                    "content_id": item["content_id"],
                    "chunk_id": item["chunk_id"],
                    "chunk_text": item["text"],
                    "similarity": float(similarity),
                    "metadata": item.get("metadata", {})
                })
            
            # Sort by similarity and take top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:top_k]
            
            return results
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def generate_response(self, query, session_id=None, language=None):
        """
        Generate a response to a user query.
        
        Args:
            query: User query
            session_id: Optional session ID for context
            language: Optional language code
            
        Returns:
            dict: Generated response
        """
        start_time = time.time()
        
        try:
            # Get session if provided
            session = None
            if session_id:
                session = self.get_session(session_id)
            
            # Determine language to use
            if not language and session:
                language = session.get("language", "en")
            elif not language:
                language = "en"
            
            # Detect language if language manager is available
            detected_language = language
            if self.language_manager:
                detection = self.language_manager.detect_language(query)
                detected_language = detection["detected_language"]
                
                # Update session language if it changed
                if session and detected_language != session.get("language"):
                    with self.sessions_lock:
                        session["language"] = detected_language
                        self._save_sessions()
            
            # Search for relevant content
            search_results = self.search_content(query, language=detected_language)
            
            # Get chat history if session provided
            chat_history = []
            if session:
                messages = self.get_messages(session_id, limit=5)
                for msg in messages:
                    role = "user" if msg["is_user"] else "assistant"
                    chat_history.append({
                        "role": role,
                        "content": msg["content"]
                    })
            
            # Generate response using LLM
            response_text, sources = self._generate_llm_response(
                query=query,
                search_results=search_results,
                chat_history=chat_history,
                language=detected_language
            )
            
            # Add response to session if provided
            if session:
                metadata = {
                    "sources": sources,
                    "language": detected_language,
                    "processing_time": time.time() - start_time
                }
                
                self.add_message(
                    session_id=session_id,
                    content=response_text,
                    is_user=False,
                    metadata=metadata
                )
            
            return {
                "response": response_text,
                "sources": sources,
                "language": detected_language,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Return error message in the appropriate language
            error_messages = {
                "en": "I'm sorry, I encountered an error while processing your request.",
                "hi": "मुझे खेद है, आपके अनुरोध को संसाधित करते समय मुझे एक त्रुटि मिली।",
                "te": "క్షమించండి, మీ అభ్యర్థనను ప్రాసెస్ చేయడంలో నాకు లోపం ఎదురైంది.",
                "ta": "மன்னிக்கவும், உங்கள் கோரிக்கையை செயலாக்கும் போது எனக்கு பிழை ஏற்பட்டது.",
                "de": "Es tut mir leid, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten."
            }
            
            error_message = error_messages.get(language, error_messages["en"])
            
            return {
                "response": error_message,
                "sources": [],
                "language": language,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _generate_llm_response(self, query, search_results, chat_history, language):
        """
        Generate a response using a language model.
        
        Args:
            query: User query
            search_results: Search results from RAG
            chat_history: Chat history
            language: Language code
            
        Returns:
            tuple: Generated response text and sources
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Determine model to use based on language
            if language == "en":
                model_name = "gpt2"
            else:
                model_name = "xlm-roberta-base"
            
            model_path = self.chat_models_path / "llm" / model_name
            
            # Check if model exists locally
            if not model_path.exists():
                logger.info(f"LLM model {model_name} not found locally, using remote model")
                model_path = model_name
            
            # Get or load model and tokenizer
            with self.model_cache_lock:
                cache_key = f"llm_{model_name}"
                
                if cache_key not in self.model_cache:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    self.model_cache[cache_key] = (model, tokenizer)
                else:
                    model, tokenizer = self.model_cache[cache_key]
            
            # Prepare context from search results
            context = ""
            sources = []
            
            for result in search_results:
                context += f"{result['chunk_text']}\n\n"
                
                # Add source if not already included
                source_id = result["content_id"]
                if source_id not in sources:
                    sources.append(source_id)
            
            # Prepare chat history
            history_text = ""
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            
            # Create prompt
            prompt = f"""
Context information:
{context}

Chat history:
{history_text}

User: {query}
Assistant:"""
            
            # Generate response
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant's response
            response = response.split("Assistant:")[-1].strip()
            
            # Translate response if needed
            if self.language_manager and language != "en":
                translation = self.language_manager.translate_text(response, source_lang="en", target_lang=language)
                response = translation["translated_text"]
            
            return response, sources
        except Exception as e:
            logger.error(f"Error in LLM response generation: {e}")
            
            # Return simple response based on language
            fallback_responses = {
                "en": "I'm sorry, I couldn't generate a proper response at this time.",
                "hi": "मुझे खेद है, मैं इस समय उचित प्रतिक्रिया नहीं दे सका।",
                "te": "క్షమించండి, నేను ఈ సమయంలో సరైన ప్రతిస్పందనను ఉత్పత్తి చేయలేకపోయాను.",
                "ta": "மன்னிக்கவும், இந்த நேரத்தில் நான் சரியான பதிலை உருவாக்க முடியவில்லை.",
                "de": "Es tut mir leid, ich konnte zu diesem Zeitpunkt keine richtige Antwort generieren."
            }
            
            fallback = fallback_responses.get(language, fallback_responses["en"])
            return fallback, []
