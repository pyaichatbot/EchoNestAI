"""
Production-Grade LLM Response Generation System
Generic, reusable LLM response generation with multiple providers
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

class ResponseType(Enum):
    """Types of responses"""
    CONVERSATIONAL = "conversational"
    INFORMATIONAL = "informational"
    INSTRUCTIONAL = "instructional"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"

@dataclass
class GenerationConfig:
    """Configuration for response generation"""
    model_name: str = "claude-3-haiku-20240307"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    response_type: ResponseType = ResponseType.CONVERSATIONAL
    system_prompt: str = ""
    enable_streaming: bool = False
    enable_function_calling: bool = False
    functions: List[Dict] = None

@dataclass
class GenerationResult:
    """Result of response generation"""
    response_text: str
    response_type: ResponseType
    confidence: float
    tokens_used: int
    cost: float
    provider: str
    model_used: str
    processing_time: float
    metadata: Dict[str, Any]

class BaseResponseGenerator(ABC):
    """Abstract base class for response generators"""
    
    @abstractmethod
    async def generate_response(self, 
                              prompt: str, 
                              context: Optional[Dict] = None,
                              config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if generator is available"""
        pass

class OpenAIResponseGenerator(BaseResponseGenerator):
    """OpenAI-based response generator"""
    
    def __init__(self, api_key: str, default_config: GenerationConfig):
        self.api_key = api_key
        self.default_config = default_config
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return bool(self.api_key)
    
    async def generate_response(self, 
                              prompt: str, 
                              context: Optional[Dict] = None,
                              config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Generate response using OpenAI"""
        start_time = time.time()
        
        try:
            # Use provided config or default
            gen_config = config or self.default_config
            
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if gen_config.system_prompt:
                messages.append({"role": "system", "content": gen_config.system_prompt})
            
            # Add context if provided
            if context:
                context_str = f"Context: {json.dumps(context)}\n\n"
                prompt = context_str + prompt
            
            messages.append({"role": "user", "content": prompt})
            
            # Prepare generation parameters
            generation_params = {
                "model": gen_config.model_name,
                "messages": messages,
                "temperature": gen_config.temperature,
                "max_tokens": gen_config.max_tokens,
                "top_p": gen_config.top_p,
                "frequency_penalty": gen_config.frequency_penalty,
                "presence_penalty": gen_config.presence_penalty
            }
            
            # Add function calling if enabled
            if gen_config.enable_function_calling and gen_config.functions:
                generation_params["functions"] = gen_config.functions
                generation_params["function_call"] = "auto"
            
            # Generate response
            response = await self.client.chat.completions.create(**generation_params)
            
            # Extract response
            response_text = response.choices[0].message.content or ""
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(response.usage.total_tokens, gen_config.model_name)
            
            processing_time = time.time() - start_time
            
            return GenerationResult(
                response_text=response_text,
                response_type=gen_config.response_type,
                confidence=0.9,  # OpenAI doesn't provide confidence scores
                tokens_used=response.usage.total_tokens,
                cost=cost,
                provider="openai",
                model_used=gen_config.model_name,
                processing_time=processing_time,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI response generation error: {e}")
            processing_time = time.time() - start_time
            
            return GenerationResult(
                response_text="I apologize, but I'm unable to generate a response at the moment. Please try again later.",
                response_type=ResponseType.CONVERSATIONAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                provider="openai",
                model_used=gen_config.model_name if config else self.default_config.model_name,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _calculate_cost(self, total_tokens: int, model_name: str) -> float:
        """Calculate approximate cost based on model and tokens"""
        # OpenAI pricing (approximate, may vary)
        pricing = {
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
            "gpt-4-turbo": 0.01 / 1000,  # $0.01 per 1K tokens
            "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
        }
        
        base_cost = pricing.get(model_name, 0.002 / 1000)
        return total_tokens * base_cost

class AnthropicResponseGenerator(BaseResponseGenerator):
    """Anthropic Claude-based response generator"""
    
    def __init__(self, api_key: str, default_config: GenerationConfig):
        self.api_key = api_key
        self.default_config = default_config
        self.client = Anthropic(api_key=api_key)
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available"""
        return bool(self.api_key)
    
    async def generate_response(self, 
                              prompt: str, 
                              context: Optional[Dict] = None,
                              config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Generate response using Anthropic Claude"""
        start_time = time.time()
        
        try:
            # Use provided config or default
            gen_config = config or self.default_config
            
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if gen_config.system_prompt:
                messages.append({"role": "system", "content": gen_config.system_prompt})
            
            # Add context if provided
            if context:
                context_str = f"Context: {json.dumps(context)}\n\n"
                prompt = context_str + prompt
            
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await self.client.messages.create(
                model=gen_config.model_name,
                max_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                messages=messages
            )
            
            # Extract response
            response_text = response.content[0].text
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(response.usage.input_tokens + response.usage.output_tokens, gen_config.model_name)
            
            processing_time = time.time() - start_time
            
            return GenerationResult(
                response_text=response_text,
                response_type=gen_config.response_type,
                confidence=0.9,  # Anthropic doesn't provide confidence scores
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                cost=cost,
                provider="anthropic",
                model_used=gen_config.model_name,
                processing_time=processing_time,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic response generation error: {e}")
            processing_time = time.time() - start_time
            
            return GenerationResult(
                response_text="I apologize, but I'm unable to generate a response at the moment. Please try again later.",
                response_type=ResponseType.CONVERSATIONAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                provider="anthropic",
                model_used=gen_config.model_name if config else self.default_config.model_name,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _calculate_cost(self, total_tokens: int, model_name: str) -> float:
        """Calculate approximate cost based on model and tokens"""
        # Anthropic pricing (approximate, may vary)
        pricing = {
            "claude-3-opus-20240229": 0.015 / 1000,  # $0.015 per 1K tokens
            "claude-3-sonnet-20240229": 0.003 / 1000,  # $0.003 per 1K tokens
            "claude-3-haiku-20240307": 0.00025 / 1000,  # $0.00025 per 1K tokens
        }
        
        base_cost = pricing.get(model_name, 0.003 / 1000)
        return total_tokens * base_cost

class LocalResponseGenerator(BaseResponseGenerator):
    """Local/offline response generator using simple templates"""
    
    def __init__(self, default_config: GenerationConfig):
        self.default_config = default_config
        
        # Simple response templates
        self.templates = {
            ResponseType.CONVERSATIONAL: [
                "I understand what you're saying about {topic}. That's an interesting point.",
                "Thank you for sharing that with me. I appreciate your perspective on {topic}.",
                "That's a great question about {topic}. Let me think about that for a moment.",
                "I see what you mean about {topic}. It's definitely something worth considering."
            ],
            ResponseType.INFORMATIONAL: [
                "Based on what I know about {topic}, here's what I can tell you.",
                "Regarding {topic}, the information I have suggests that...",
                "Let me share what I understand about {topic}.",
                "Here's what I can tell you about {topic}."
            ],
            ResponseType.INSTRUCTIONAL: [
                "To help you with {topic}, here's what you can do:",
                "For {topic}, I recommend the following steps:",
                "Here's how you can approach {topic}:",
                "To address {topic}, consider these actions:"
            ],
            ResponseType.CREATIVE: [
                "That's a creative idea about {topic}! It makes me think of...",
                "What an interesting perspective on {topic}. It reminds me of...",
                "That's a unique way to look at {topic}. It could lead to...",
                "Your thoughts on {topic} are quite imaginative. Perhaps..."
            ],
            ResponseType.ANALYTICAL: [
                "Looking at {topic} analytically, I can see several factors:",
                "From an analytical perspective on {topic}, we should consider:",
                "Analyzing {topic}, I notice these key elements:",
                "When we examine {topic} closely, we find that..."
            ]
        }
    
    def is_available(self) -> bool:
        """Local generator is always available"""
        return True
    
    async def generate_response(self, 
                              prompt: str, 
                              context: Optional[Dict] = None,
                              config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Generate response using local templates"""
        start_time = time.time()
        
        try:
            # Use provided config or default
            gen_config = config or self.default_config
            
            # Extract topic from prompt (simple keyword extraction)
            topic = self._extract_topic(prompt)
            
            # Get appropriate templates
            templates = self.templates.get(gen_config.response_type, self.templates[ResponseType.CONVERSATIONAL])
            
            # Select template based on prompt content
            import random
            selected_template = random.choice(templates)
            
            # Fill template
            response_text = selected_template.format(topic=topic or "this topic")
            
            # Add some variation based on prompt
            if "?" in prompt:
                response_text += " Would you like me to elaborate on that?"
            elif "thank" in prompt.lower():
                response_text = "You're welcome! I'm happy to help."
            elif "hello" in prompt.lower() or "hi" in prompt.lower():
                response_text = "Hello! How can I assist you today?"
            
            processing_time = time.time() - start_time
            
            return GenerationResult(
                response_text=response_text,
                response_type=gen_config.response_type,
                confidence=0.6,  # Lower confidence for template-based responses
                tokens_used=len(response_text.split()),
                cost=0.0,  # No cost for local generation
                provider="local",
                model_used="template_based",
                processing_time=processing_time,
                metadata={"method": "template_based", "topic": topic}
            )
            
        except Exception as e:
            logger.error(f"Local response generation error: {e}")
            processing_time = time.time() - start_time
            
            return GenerationResult(
                response_text="I'm sorry, but I'm having trouble generating a response right now.",
                response_type=ResponseType.CONVERSATIONAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                provider="local",
                model_used="template_based",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _extract_topic(self, prompt: str) -> str:
        """Extract topic from prompt using simple keyword matching"""
        # Simple keyword extraction
        keywords = [
            "weather", "time", "date", "help", "question", "problem", "issue",
            "work", "study", "learn", "teach", "explain", "understand",
            "music", "movie", "book", "food", "travel", "health", "exercise"
        ]
        
        prompt_lower = prompt.lower()
        for keyword in keywords:
            if keyword in prompt_lower:
                return keyword
        
        return "this topic"

class ResponseGenerationEngine:
    """Production-grade response generation engine with multiple providers"""
    
    def __init__(self, generators: List[BaseResponseGenerator], default_config: GenerationConfig):
        self.generators = generators
        self.default_config = default_config
        self.primary_generator = None
        self.fallback_generator = None
        
        # Set up primary and fallback generators
        self._setup_generators()
    
    def _setup_generators(self):
        """Set up primary and fallback generators"""
        available_generators = [g for g in self.generators if g.is_available()]
        
        if not available_generators:
            raise RuntimeError("No response generators are available")
        
        # Set primary generator (first available)
        self.primary_generator = available_generators[0]
        
        # Set fallback generator (local if available, otherwise first available)
        fallback_candidates = [g for g in available_generators if isinstance(g, LocalResponseGenerator)]
        if fallback_candidates:
            self.fallback_generator = fallback_candidates[0]
        else:
            self.fallback_generator = available_generators[0]
        
        logger.info(f"Response generation engine initialized with {len(available_generators)} generators")
        logger.info(f"Primary generator: {self.primary_generator.__class__.__name__}")
        logger.info(f"Fallback generator: {self.fallback_generator.__class__.__name__}")
    
    async def generate_response(self, 
                              prompt: str, 
                              context: Optional[Dict] = None,
                              config: Optional[GenerationConfig] = None,
                              use_fallback: bool = True) -> GenerationResult:
        """
        Generate response using primary generator with fallback
        
        Args:
            prompt: Input prompt
            context: Optional context information
            config: Generation configuration
            use_fallback: Whether to use fallback generator on failure
            
        Returns:
            GenerationResult object with generated response
        """
        if not prompt or not prompt.strip():
            return GenerationResult(
                response_text="I didn't receive any input. Could you please provide a prompt?",
                response_type=ResponseType.CONVERSATIONAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                provider="none",
                model_used="none",
                processing_time=0.0,
                metadata={"note": "Empty prompt"}
            )
        
        # Try primary generator
        try:
            result = await self.primary_generator.generate_response(prompt, context, config)
            
            # Check if response is valid
            if result.response_text and len(result.response_text.strip()) > 0:
                return result
            
            # If response is empty, try fallback
            if use_fallback and self.fallback_generator != self.primary_generator:
                logger.info("Primary generator returned empty response, trying fallback")
                fallback_result = await self.fallback_generator.generate_response(prompt, context, config)
                
                if fallback_result.response_text and len(fallback_result.response_text.strip()) > 0:
                    return fallback_result
            
            return result
            
        except Exception as e:
            logger.error(f"Primary response generator failed: {e}")
            
            if use_fallback and self.fallback_generator != self.primary_generator:
                try:
                    logger.info("Using fallback response generator")
                    return await self.fallback_generator.generate_response(prompt, context, config)
                except Exception as fallback_error:
                    logger.error(f"Fallback response generator also failed: {fallback_error}")
            
            # Return error response
            return GenerationResult(
                response_text="I apologize, but I'm unable to generate a response at the moment. Please try again later.",
                response_type=ResponseType.CONVERSATIONAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                provider="error",
                model_used="none",
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_available_generators(self) -> List[str]:
        """Get list of available generator names"""
        return [g.__class__.__name__ for g in self.generators if g.is_available()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "total_generators": len(self.generators),
            "available_generators": len([g for g in self.generators if g.is_available()]),
            "primary_generator": self.primary_generator.__class__.__name__ if self.primary_generator else None,
            "fallback_generator": self.fallback_generator.__class__.__name__ if self.fallback_generator else None,
            "default_model": self.default_config.model_name
        } 