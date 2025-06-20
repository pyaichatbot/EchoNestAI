"""
Production-Grade Guard Rails System
Generic, reusable safety, moderation, and filtering system
"""

import json
import logging
import time
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import aiohttp
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for content"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    BLOCKED = "blocked"

class ContentType(Enum):
    """Types of content to moderate"""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

@dataclass
class SafetyResult:
    """Result of safety check"""
    is_safe: bool
    safety_level: SafetyLevel
    risk_score: float
    flagged_categories: List[str]
    flagged_content: List[str]
    recommendations: List[str]
    content_type: ContentType
    provider: str
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class GuardRailsConfig:
    """Configuration for guard rails"""
    safety_threshold: float = 0.7
    enable_content_filtering: bool = True
    enable_toxicity_detection: bool = True
    enable_bias_detection: bool = True
    enable_pii_detection: bool = True
    enable_harmful_content_detection: bool = True
    blocked_keywords: List[str] = None
    allowed_domains: List[str] = None
    max_content_length: int = 10000
    enable_llm_moderation: bool = True
    fallback_to_rules: bool = True

class BaseGuardRails(ABC):
    """Abstract base class for guard rails"""
    
    @abstractmethod
    async def check_safety(self, content: str, content_type: ContentType = ContentType.TEXT) -> SafetyResult:
        """Check content safety"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if guard rails is available"""
        pass

class OpenAIModerationGuardRails(BaseGuardRails):
    """OpenAI-based content moderation"""
    
    def __init__(self, api_key: str, config: GuardRailsConfig):
        self.api_key = api_key
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # OpenAI moderation categories
        self.categories = {
            "hate": "Content expressing hate or violence",
            "hate/threatening": "Hateful content with threats",
            "self-harm": "Content promoting self-harm",
            "sexual": "Sexual content",
            "sexual/minors": "Sexual content involving minors",
            "violence": "Violent content",
            "violence/graphic": "Graphically violent content"
        }
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return bool(self.api_key)
    
    async def check_safety(self, content: str, content_type: ContentType = ContentType.TEXT) -> SafetyResult:
        """Check content safety using OpenAI moderation"""
        start_time = time.time()
        
        try:
            # Use OpenAI moderation API
            response = await self.client.moderations.create(input=content)
            result = response.results[0]
            
            # Process results
            categories = result.categories
            category_scores = result.category_scores
            
            # Determine safety level
            flagged_categories = []
            risk_score = 0.0
            
            for category, is_flagged in categories.__dict__.items():
                if is_flagged:
                    flagged_categories.append(category)
                    risk_score = max(risk_score, getattr(category_scores, category, 0.0))
            
            # Determine safety level based on risk score
            if risk_score >= 0.9:
                safety_level = SafetyLevel.BLOCKED
                is_safe = False
            elif risk_score >= 0.7:
                safety_level = SafetyLevel.HIGH_RISK
                is_safe = False
            elif risk_score >= 0.5:
                safety_level = SafetyLevel.MEDIUM_RISK
                is_safe = False
            elif risk_score >= 0.3:
                safety_level = SafetyLevel.LOW_RISK
                is_safe = True
            else:
                safety_level = SafetyLevel.SAFE
                is_safe = True
            
            # Generate recommendations
            recommendations = self._generate_recommendations(flagged_categories, risk_score)
            
            processing_time = time.time() - start_time
            
            return SafetyResult(
                is_safe=is_safe,
                safety_level=safety_level,
                risk_score=risk_score,
                flagged_categories=flagged_categories,
                flagged_content=[content] if not is_safe else [],
                recommendations=recommendations,
                content_type=content_type,
                provider="openai",
                processing_time=processing_time,
                metadata={
                    "categories": asdict(categories),
                    "category_scores": asdict(category_scores),
                    "flagged": result.flagged
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI moderation error: {e}")
            processing_time = time.time() - start_time
            
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.HIGH_RISK,
                risk_score=1.0,
                flagged_categories=["error"],
                flagged_content=[content],
                recommendations=["Content moderation failed, blocking for safety"],
                content_type=content_type,
                provider="openai",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _generate_recommendations(self, flagged_categories: List[str], risk_score: float) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if "hate" in flagged_categories:
            recommendations.append("Content contains hate speech - consider removing or rephrasing")
        if "violence" in flagged_categories:
            recommendations.append("Content contains violent language - consider using more neutral terms")
        if "self-harm" in flagged_categories:
            recommendations.append("Content promotes self-harm - this is not allowed")
        if "sexual" in flagged_categories:
            recommendations.append("Content contains inappropriate sexual content")
        
        if risk_score > 0.8:
            recommendations.append("High risk content detected - immediate action recommended")
        elif risk_score > 0.6:
            recommendations.append("Moderate risk content - review recommended")
        
        return recommendations

class AnthropicGuardRails(BaseGuardRails):
    """Anthropic Claude-based content moderation"""
    
    def __init__(self, api_key: str, config: GuardRailsConfig):
        self.api_key = api_key
        self.config = config
        self.client = Anthropic(api_key=api_key)
        
        self.moderation_prompt = self._create_moderation_prompt()
    
    def _create_moderation_prompt(self) -> str:
        """Create moderation prompt for Claude"""
        return """
You are a content safety moderator. Analyze the given text for safety concerns.

Check for the following categories:
1. Hate speech and discrimination
2. Violence and threats
3. Self-harm content
4. Sexual content (especially involving minors)
5. Harassment and bullying
6. Misinformation and fake news
7. Personal information (PII)
8. Inappropriate language

Provide a JSON response with:
{
    "is_safe": true/false,
    "risk_score": 0.0-1.0,
    "flagged_categories": ["category1", "category2"],
    "flagged_content": ["specific problematic phrases"],
    "recommendations": ["action items"],
    "reasoning": "brief explanation"
}

Analyze the following content:
"""
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available"""
        return bool(self.api_key)
    
    async def check_safety(self, content: str, content_type: ContentType = ContentType.TEXT) -> SafetyResult:
        """Check content safety using Anthropic Claude"""
        start_time = time.time()
        
        try:
            full_prompt = self.moderation_prompt + f"\nContent: {content}\n"
            
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Parse response
            content_text = response.content[0].text.strip()
            
            # Extract JSON
            json_start = content_text.find('{')
            json_end = content_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = content_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
            
            # Process results
            is_safe = result.get("is_safe", False)
            risk_score = float(result.get("risk_score", 1.0))
            flagged_categories = result.get("flagged_categories", [])
            flagged_content = result.get("flagged_content", [])
            recommendations = result.get("recommendations", [])
            reasoning = result.get("reasoning", "")
            
            # Determine safety level
            if risk_score >= 0.9:
                safety_level = SafetyLevel.BLOCKED
            elif risk_score >= 0.7:
                safety_level = SafetyLevel.HIGH_RISK
            elif risk_score >= 0.5:
                safety_level = SafetyLevel.MEDIUM_RISK
            elif risk_score >= 0.3:
                safety_level = SafetyLevel.LOW_RISK
            else:
                safety_level = SafetyLevel.SAFE
            
            processing_time = time.time() - start_time
            
            return SafetyResult(
                is_safe=is_safe,
                safety_level=safety_level,
                risk_score=risk_score,
                flagged_categories=flagged_categories,
                flagged_content=flagged_content,
                recommendations=recommendations,
                content_type=content_type,
                provider="anthropic",
                processing_time=processing_time,
                metadata={"reasoning": reasoning}
            )
            
        except Exception as e:
            logger.error(f"Anthropic moderation error: {e}")
            processing_time = time.time() - start_time
            
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.HIGH_RISK,
                risk_score=1.0,
                flagged_categories=["error"],
                flagged_content=[content],
                recommendations=["Content moderation failed, blocking for safety"],
                content_type=content_type,
                provider="anthropic",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

class RuleBasedGuardRails(BaseGuardRails):
    """Rule-based content filtering as fallback"""
    
    def __init__(self, config: GuardRailsConfig):
        self.config = config
        
        # Define patterns for different types of content
        self.hate_patterns = [
            r'\b(kill|murder|death|hate|racist|sexist|homophobic|transphobic)\b',
            r'\b(nazi|hitler|white\s+supremacy|black\s+lives\s+dont\s+matter)\b'
        ]
        
        self.violence_patterns = [
            r'\b(bomb|explode|shoot|gun|weapon|attack|fight|war)\b',
            r'\b(destroy|burn|kill|murder|assassinate)\b'
        ]
        
        self.self_harm_patterns = [
            r'\b(suicide|kill\s+myself|end\s+it\s+all|want\s+to\s+die)\b',
            r'\b(cut\s+myself|self\s+harm|hurt\s+myself)\b'
        ]
        
        self.sexual_patterns = [
            r'\b(sex|porn|nude|naked|penis|vagina|fuck|dick|pussy)\b',
            r'\b(child\s+porn|pedo|pedophile|underage)\b'
        ]
        
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'  # IP address
        ]
        
        # Compile patterns
        import re
        self.hate_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.hate_patterns]
        self.violence_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.violence_patterns]
        self.self_harm_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.self_harm_patterns]
        self.sexual_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.sexual_patterns]
        self.pii_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.pii_patterns]
        
        # Blocked keywords
        self.blocked_keywords = set(self.config.blocked_keywords or [])
    
    def is_available(self) -> bool:
        """Rule-based guard rails is always available"""
        return True
    
    async def check_safety(self, content: str, content_type: ContentType = ContentType.TEXT) -> SafetyResult:
        """Check content safety using rule-based patterns"""
        start_time = time.time()
        
        try:
            content_lower = content.lower()
            flagged_categories = []
            flagged_content = []
            risk_score = 0.0
            
            # Check for blocked keywords
            for keyword in self.blocked_keywords:
                if keyword.lower() in content_lower:
                    flagged_categories.append("blocked_keyword")
                    flagged_content.append(keyword)
                    risk_score = max(risk_score, 0.8)
            
            # Check for hate speech
            if any(regex.search(content) for regex in self.hate_regex):
                flagged_categories.append("hate")
                flagged_content.extend(self._extract_matches(content, self.hate_regex))
                risk_score = max(risk_score, 0.9)
            
            # Check for violence
            if any(regex.search(content) for regex in self.violence_regex):
                flagged_categories.append("violence")
                flagged_content.extend(self._extract_matches(content, self.violence_regex))
                risk_score = max(risk_score, 0.8)
            
            # Check for self-harm
            if any(regex.search(content) for regex in self.self_harm_regex):
                flagged_categories.append("self_harm")
                flagged_content.extend(self._extract_matches(content, self.self_harm_regex))
                risk_score = max(risk_score, 0.95)
            
            # Check for sexual content
            if any(regex.search(content) for regex in self.sexual_regex):
                flagged_categories.append("sexual")
                flagged_content.extend(self._extract_matches(content, self.sexual_regex))
                risk_score = max(risk_score, 0.9)
            
            # Check for PII
            if any(regex.search(content) for regex in self.pii_regex):
                flagged_categories.append("pii")
                flagged_content.extend(self._extract_matches(content, self.pii_regex))
                risk_score = max(risk_score, 0.7)
            
            # Determine safety level
            if risk_score >= 0.9:
                safety_level = SafetyLevel.BLOCKED
                is_safe = False
            elif risk_score >= 0.7:
                safety_level = SafetyLevel.HIGH_RISK
                is_safe = False
            elif risk_score >= 0.5:
                safety_level = SafetyLevel.MEDIUM_RISK
                is_safe = False
            elif risk_score >= 0.3:
                safety_level = SafetyLevel.LOW_RISK
                is_safe = True
            else:
                safety_level = SafetyLevel.SAFE
                is_safe = True
            
            # Generate recommendations
            recommendations = self._generate_recommendations(flagged_categories, risk_score)
            
            processing_time = time.time() - start_time
            
            return SafetyResult(
                is_safe=is_safe,
                safety_level=safety_level,
                risk_score=risk_score,
                flagged_categories=flagged_categories,
                flagged_content=flagged_content,
                recommendations=recommendations,
                content_type=content_type,
                provider="rule_based",
                processing_time=processing_time,
                metadata={"method": "pattern_matching"}
            )
            
        except Exception as e:
            logger.error(f"Rule-based guard rails error: {e}")
            processing_time = time.time() - start_time
            
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.HIGH_RISK,
                risk_score=1.0,
                flagged_categories=["error"],
                flagged_content=[content],
                recommendations=["Content safety check failed, blocking for safety"],
                content_type=content_type,
                provider="rule_based",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _extract_matches(self, content: str, regex_list: List) -> List[str]:
        """Extract matching content from regex patterns"""
        matches = []
        for regex in regex_list:
            found = regex.findall(content)
            matches.extend(found)
        return list(set(matches))  # Remove duplicates
    
    def _generate_recommendations(self, flagged_categories: List[str], risk_score: float) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if "hate" in flagged_categories:
            recommendations.append("Content contains hate speech - consider removing or rephrasing")
        if "violence" in flagged_categories:
            recommendations.append("Content contains violent language - consider using more neutral terms")
        if "self_harm" in flagged_categories:
            recommendations.append("Content promotes self-harm - this is not allowed")
        if "sexual" in flagged_categories:
            recommendations.append("Content contains inappropriate sexual content")
        if "pii" in flagged_categories:
            recommendations.append("Content contains personal information - remove for privacy")
        if "blocked_keyword" in flagged_categories:
            recommendations.append("Content contains blocked keywords - review and remove")
        
        if risk_score > 0.8:
            recommendations.append("High risk content detected - immediate action recommended")
        elif risk_score > 0.6:
            recommendations.append("Moderate risk content - review recommended")
        
        return recommendations

class GuardRailsEngine:
    """Production-grade guard rails engine with multiple providers"""
    
    def __init__(self, guard_rails: List[BaseGuardRails], config: GuardRailsConfig):
        self.guard_rails = guard_rails
        self.config = config
        self.primary_guard_rails = None
        self.fallback_guard_rails = None
        
        # Set up primary and fallback guard rails
        self._setup_guard_rails()
    
    def _setup_guard_rails(self):
        """Set up primary and fallback guard rails"""
        available_guard_rails = [g for g in self.guard_rails if g.is_available()]
        
        if not available_guard_rails:
            raise RuntimeError("No guard rails are available")
        
        # Set primary guard rails (first available)
        self.primary_guard_rails = available_guard_rails[0]
        
        # Set fallback guard rails (rule-based if available, otherwise first available)
        fallback_candidates = [g for g in available_guard_rails if isinstance(g, RuleBasedGuardRails)]
        if fallback_candidates:
            self.fallback_guard_rails = fallback_candidates[0]
        else:
            self.fallback_guard_rails = available_guard_rails[0]
        
        logger.info(f"Guard rails engine initialized with {len(available_guard_rails)} providers")
        logger.info(f"Primary guard rails: {self.primary_guard_rails.__class__.__name__}")
        logger.info(f"Fallback guard rails: {self.fallback_guard_rails.__class__.__name__}")
    
    async def check_safety(self, content: str, content_type: ContentType = ContentType.TEXT, use_fallback: bool = True) -> SafetyResult:
        """
        Check content safety using primary guard rails with fallback
        
        Args:
            content: Content to check
            content_type: Type of content
            use_fallback: Whether to use fallback guard rails on failure
            
        Returns:
            SafetyResult object with safety assessment
        """
        if not content or not content.strip():
            return SafetyResult(
                is_safe=True,
                safety_level=SafetyLevel.SAFE,
                risk_score=0.0,
                flagged_categories=[],
                flagged_content=[],
                recommendations=[],
                content_type=content_type,
                provider="none",
                processing_time=0.0,
                metadata={"note": "Empty content"}
            )
        
        # Check content length
        if len(content) > self.config.max_content_length:
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                risk_score=1.0,
                flagged_categories=["content_too_long"],
                flagged_content=[content[:100] + "..."],
                recommendations=[f"Content exceeds maximum length of {self.config.max_content_length} characters"],
                content_type=content_type,
                provider="length_check",
                processing_time=0.0,
                metadata={"content_length": len(content)}
            )
        
        # Try primary guard rails
        try:
            result = await self.primary_guard_rails.check_safety(content, content_type)
            
            # Check if result meets safety threshold
            if result.risk_score <= self.config.safety_threshold:
                return result
            
            # If risk is high, try fallback
            if use_fallback and self.fallback_guard_rails != self.primary_guard_rails:
                logger.info(f"Primary guard rails risk too high ({result.risk_score}), trying fallback")
                fallback_result = await self.fallback_guard_rails.check_safety(content, content_type)
                
                # Use fallback if it has lower risk score
                if fallback_result.risk_score < result.risk_score:
                    return fallback_result
            
            return result
            
        except Exception as e:
            logger.error(f"Primary guard rails failed: {e}")
            
            if use_fallback and self.fallback_guard_rails != self.primary_guard_rails:
                try:
                    logger.info("Using fallback guard rails")
                    return await self.fallback_guard_rails.check_safety(content, content_type)
                except Exception as fallback_error:
                    logger.error(f"Fallback guard rails also failed: {fallback_error}")
            
            # Return blocked result on error
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                risk_score=1.0,
                flagged_categories=["error"],
                flagged_content=[content],
                recommendations=["Content safety check failed, blocking for safety"],
                content_type=content_type,
                provider="error",
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_available_guard_rails(self) -> List[str]:
        """Get list of available guard rails names"""
        return [g.__class__.__name__ for g in self.guard_rails if g.is_available()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "total_guard_rails": len(self.guard_rails),
            "available_guard_rails": len([g for g in self.guard_rails if g.is_available()]),
            "primary_guard_rails": self.primary_guard_rails.__class__.__name__ if self.primary_guard_rails else None,
            "fallback_guard_rails": self.fallback_guard_rails.__class__.__name__ if self.fallback_guard_rails else None,
            "safety_threshold": self.config.safety_threshold
        } 