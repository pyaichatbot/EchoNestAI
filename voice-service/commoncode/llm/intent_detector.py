"""
Production-Grade LLM-based Intent Detection System
Generic, reusable intent detection using multiple LLM providers
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import aiohttp
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Common intent types"""
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    STATEMENT = "statement"
    CLARIFICATION = "clarification"
    GOODBYE = "goodbye"
    HELP = "help"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class Intent:
    """Intent detection result"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    slots: Dict[str, Any]
    raw_text: str
    processed_text: str
    language: str
    provider: str
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class IntentConfig:
    """Configuration for intent detection"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 150
    confidence_threshold: float = 0.7
    language_detection: bool = True
    entity_extraction: bool = True
    slot_filling: bool = True
    custom_intents: List[str] = None
    fallback_intent: IntentType = IntentType.UNKNOWN

class BaseIntentDetector(ABC):
    """Abstract base class for intent detectors"""
    
    @abstractmethod
    async def detect_intent(self, text: str, context: Optional[Dict] = None) -> Intent:
        """Detect intent from text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if detector is available"""
        pass

class OpenAIIntentDetector(BaseIntentDetector):
    """OpenAI-based intent detector"""
    
    def __init__(self, api_key: str, config: IntentConfig):
        self.api_key = api_key
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Define intent detection prompt
        self.intent_prompt = self._create_intent_prompt()
    
    def _create_intent_prompt(self) -> str:
        """Create the intent detection prompt"""
        base_intents = [intent.value for intent in IntentType]
        custom_intents = self.config.custom_intents or []
        all_intents = base_intents + custom_intents
        
        prompt = f"""
You are an intent detection system. Analyze the given text and determine the user's intent.

Available intents: {', '.join(all_intents)}

For each text, provide a JSON response with the following structure:
{{
    "intent": "intent_type",
    "confidence": 0.95,
    "entities": {{"entity_type": "value"}},
    "slots": {{"slot_name": "value"}},
    "language": "en",
    "reasoning": "brief explanation"
}}

Intent definitions:
- greeting: User is greeting or saying hello
- question: User is asking a question
- command: User is giving a command or instruction
- statement: User is making a statement or declaration
- clarification: User is asking for clarification
- goodbye: User is saying goodbye or ending conversation
- help: User is asking for help
- error: User is reporting an error or problem
- unknown: Intent cannot be determined

Entity types to extract:
- person: Names of people
- location: Places, addresses
- time: Dates, times, durations
- number: Quantities, amounts
- action: Verbs, actions
- object: Nouns, things

Analyze the following text:
"""
        return prompt
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return bool(self.api_key)
    
    async def detect_intent(self, text: str, context: Optional[Dict] = None) -> Intent:
        """Detect intent using OpenAI"""
        start_time = time.time()
        
        try:
            # Prepare context for the prompt
            context_str = ""
            if context:
                context_str = f"\nContext: {json.dumps(context)}\n"
            
            full_prompt = self.intent_prompt + context_str + f"Text: {text}\n"
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an intent detection system. Respond only with valid JSON."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
            
            # Process result
            intent_type = IntentType(result.get("intent", "unknown"))
            confidence = float(result.get("confidence", 0.0))
            entities = result.get("entities", {})
            slots = result.get("slots", {})
            language = result.get("language", "en")
            reasoning = result.get("reasoning", "")
            
            processing_time = time.time() - start_time
            
            return Intent(
                intent_type=intent_type,
                confidence=confidence,
                entities=entities,
                slots=slots,
                raw_text=text,
                processed_text=text,
                language=language,
                provider="openai",
                processing_time=processing_time,
                metadata={"reasoning": reasoning, "model": self.config.model_name}
            )
            
        except Exception as e:
            logger.error(f"OpenAI intent detection error: {e}")
            processing_time = time.time() - start_time
            
            return Intent(
                intent_type=self.config.fallback_intent,
                confidence=0.0,
                entities={},
                slots={},
                raw_text=text,
                processed_text=text,
                language="en",
                provider="openai",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

class AnthropicIntentDetector(BaseIntentDetector):
    """Anthropic Claude-based intent detector"""
    
    def __init__(self, api_key: str, config: IntentConfig):
        self.api_key = api_key
        self.config = config
        self.client = Anthropic(api_key=api_key)
        
        # Define intent detection prompt
        self.intent_prompt = self._create_intent_prompt()
    
    def _create_intent_prompt(self) -> str:
        """Create the intent detection prompt for Claude"""
        base_intents = [intent.value for intent in IntentType]
        custom_intents = self.config.custom_intents or []
        all_intents = base_intents + custom_intents
        
        prompt = f"""
You are an intent detection system. Analyze the given text and determine the user's intent.

Available intents: {', '.join(all_intents)}

For each text, provide a JSON response with the following structure:
{{
    "intent": "intent_type",
    "confidence": 0.95,
    "entities": {{"entity_type": "value"}},
    "slots": {{"slot_name": "value"}},
    "language": "en",
    "reasoning": "brief explanation"
}}

Intent definitions:
- greeting: User is greeting or saying hello
- question: User is asking a question
- command: User is giving a command or instruction
- statement: User is making a statement or declaration
- clarification: User is asking for clarification
- goodbye: User is saying goodbye or ending conversation
- help: User is asking for help
- error: User is reporting an error or problem
- unknown: Intent cannot be determined

Entity types to extract:
- person: Names of people
- location: Places, addresses
- time: Dates, times, durations
- number: Quantities, amounts
- action: Verbs, actions
- object: Nouns, things

Analyze the following text:
"""
        return prompt
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available"""
        return bool(self.api_key)
    
    async def detect_intent(self, text: str, context: Optional[Dict] = None) -> Intent:
        """Detect intent using Anthropic Claude"""
        start_time = time.time()
        
        try:
            # Prepare context for the prompt
            context_str = ""
            if context:
                context_str = f"\nContext: {json.dumps(context)}\n"
            
            full_prompt = self.intent_prompt + context_str + f"Text: {text}\n"
            
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Parse response
            content = response.content[0].text.strip()
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
            
            # Process result
            intent_type = IntentType(result.get("intent", "unknown"))
            confidence = float(result.get("confidence", 0.0))
            entities = result.get("entities", {})
            slots = result.get("slots", {})
            language = result.get("language", "en")
            reasoning = result.get("reasoning", "")
            
            processing_time = time.time() - start_time
            
            return Intent(
                intent_type=intent_type,
                confidence=confidence,
                entities=entities,
                slots=slots,
                raw_text=text,
                processed_text=text,
                language=language,
                provider="anthropic",
                processing_time=processing_time,
                metadata={"reasoning": reasoning, "model": self.config.model_name}
            )
            
        except Exception as e:
            logger.error(f"Anthropic intent detection error: {e}")
            processing_time = time.time() - start_time
            
            return Intent(
                intent_type=self.config.fallback_intent,
                confidence=0.0,
                entities={},
                slots={},
                raw_text=text,
                processed_text=text,
                language="en",
                provider="anthropic",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

class RuleBasedIntentDetector(BaseIntentDetector):
    """Rule-based intent detector as fallback"""
    
    def __init__(self, config: IntentConfig):
        self.config = config
        self.greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
            r'\b(namaste|namaskar|salaam|shalom)\b'
        ]
        self.question_patterns = [
            r'\b(what|when|where|who|why|how|which|can|could|would|will|do|does|did|is|are|was|were)\b.*\?',
            r'\?$'
        ]
        self.command_patterns = [
            r'\b(play|stop|pause|resume|next|previous|volume|mute|unmute)\b',
            r'\b(open|close|start|end|begin|finish)\b',
            r'\b(save|delete|create|edit|update|modify)\b'
        ]
        self.goodbye_patterns = [
            r'\b(bye|goodbye|see you|farewell|take care)\b',
            r'\b(thank you|thanks)\b'
        ]
        self.help_patterns = [
            r'\b(help|support|assist|guide|explain|show)\b'
        ]
        
        import re
        self.greeting_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.greeting_patterns]
        self.question_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.question_patterns]
        self.command_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.command_patterns]
        self.goodbye_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.goodbye_patterns]
        self.help_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.help_patterns]
    
    def is_available(self) -> bool:
        """Rule-based detector is always available"""
        return True
    
    async def detect_intent(self, text: str, context: Optional[Dict] = None) -> Intent:
        """Detect intent using rule-based patterns"""
        start_time = time.time()
        
        try:
            text_lower = text.lower().strip()
            
            # Check patterns in order of specificity
            if any(regex.search(text) for regex in self.help_regex):
                intent_type = IntentType.HELP
                confidence = 0.8
            elif any(regex.search(text) for regex in self.goodbye_regex):
                intent_type = IntentType.GOODBYE
                confidence = 0.9
            elif any(regex.search(text) for regex in self.command_regex):
                intent_type = IntentType.COMMAND
                confidence = 0.85
            elif any(regex.search(text) for regex in self.question_regex):
                intent_type = IntentType.QUESTION
                confidence = 0.9
            elif any(regex.search(text) for regex in self.greeting_regex):
                intent_type = IntentType.GREETING
                confidence = 0.95
            else:
                intent_type = IntentType.STATEMENT
                confidence = 0.6
            
            # Simple entity extraction
            entities = self._extract_entities(text)
            slots = self._extract_slots(text)
            
            processing_time = time.time() - start_time
            
            return Intent(
                intent_type=intent_type,
                confidence=confidence,
                entities=entities,
                slots=slots,
                raw_text=text,
                processed_text=text,
                language="en",  # Rule-based doesn't detect language
                provider="rule_based",
                processing_time=processing_time,
                metadata={"method": "pattern_matching"}
            )
            
        except Exception as e:
            logger.error(f"Rule-based intent detection error: {e}")
            processing_time = time.time() - start_time
            
            return Intent(
                intent_type=self.config.fallback_intent,
                confidence=0.0,
                entities={},
                slots={},
                raw_text=text,
                processed_text=text,
                language="en",
                provider="rule_based",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract basic entities using patterns"""
        entities = {}
        
        # Extract numbers
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            entities["number"] = numbers
        
        # Extract time patterns
        time_patterns = re.findall(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', text)
        if time_patterns:
            entities["time"] = time_patterns
        
        return entities
    
    def _extract_slots(self, text: str) -> Dict[str, Any]:
        """Extract basic slots"""
        slots = {}
        
        # Extract action words
        import re
        action_words = re.findall(r'\b(play|stop|pause|resume|next|previous|volume|mute|unmute|open|close|start|end)\b', text.lower())
        if action_words:
            slots["action"] = action_words[0]
        
        return slots

class IntentDetectionEngine:
    """Production-grade intent detection engine with multiple providers"""
    
    def __init__(self, detectors: List[BaseIntentDetector], config: IntentConfig):
        self.detectors = detectors
        self.config = config
        self.primary_detector = None
        self.fallback_detector = None
        
        # Set up primary and fallback detectors
        self._setup_detectors()
    
    def _setup_detectors(self):
        """Set up primary and fallback detectors"""
        available_detectors = [d for d in self.detectors if d.is_available()]
        
        if not available_detectors:
            raise RuntimeError("No intent detectors are available")
        
        # Set primary detector (first available)
        self.primary_detector = available_detectors[0]
        
        # Set fallback detector (rule-based if available, otherwise first available)
        fallback_candidates = [d for d in available_detectors if isinstance(d, RuleBasedIntentDetector)]
        if fallback_candidates:
            self.fallback_detector = fallback_candidates[0]
        else:
            self.fallback_detector = available_detectors[0]
        
        logger.info(f"Intent detection engine initialized with {len(available_detectors)} detectors")
        logger.info(f"Primary detector: {self.primary_detector.__class__.__name__}")
        logger.info(f"Fallback detector: {self.fallback_detector.__class__.__name__}")
    
    async def detect_intent(self, text: str, context: Optional[Dict] = None, use_fallback: bool = True) -> Intent:
        """
        Detect intent using primary detector with fallback
        
        Args:
            text: Text to analyze
            context: Optional context information
            use_fallback: Whether to use fallback detector on failure
            
        Returns:
            Intent object with detection results
        """
        if not text or not text.strip():
            return Intent(
                intent_type=self.config.fallback_intent,
                confidence=0.0,
                entities={},
                slots={},
                raw_text=text or "",
                processed_text=text or "",
                language="en",
                provider="none",
                processing_time=0.0,
                metadata={"error": "Empty text"}
            )
        
        # Try primary detector
        try:
            intent = await self.primary_detector.detect_intent(text, context)
            
            # Check confidence threshold
            if intent.confidence >= self.config.confidence_threshold:
                return intent
            
            # If confidence is low, try fallback
            if use_fallback and self.fallback_detector != self.primary_detector:
                logger.info(f"Primary detector confidence too low ({intent.confidence}), trying fallback")
                fallback_intent = await self.fallback_detector.detect_intent(text, context)
                
                # Use fallback if it has higher confidence
                if fallback_intent.confidence > intent.confidence:
                    return fallback_intent
            
            return intent
            
        except Exception as e:
            logger.error(f"Primary intent detector failed: {e}")
            
            if use_fallback and self.fallback_detector != self.primary_detector:
                try:
                    logger.info("Using fallback intent detector")
                    return await self.fallback_detector.detect_intent(text, context)
                except Exception as fallback_error:
                    logger.error(f"Fallback intent detector also failed: {fallback_error}")
            
            # Return error intent
            return Intent(
                intent_type=self.config.fallback_intent,
                confidence=0.0,
                entities={},
                slots={},
                raw_text=text,
                processed_text=text,
                language="en",
                provider="error",
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detector names"""
        return [d.__class__.__name__ for d in self.detectors if d.is_available()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "total_detectors": len(self.detectors),
            "available_detectors": len([d for d in self.detectors if d.is_available()]),
            "primary_detector": self.primary_detector.__class__.__name__ if self.primary_detector else None,
            "fallback_detector": self.fallback_detector.__class__.__name__ if self.fallback_detector else None,
            "confidence_threshold": self.config.confidence_threshold
        } 