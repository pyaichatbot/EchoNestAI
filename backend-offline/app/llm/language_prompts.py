from typing import Dict, Any, Optional, List
import os

# Language-specific prompts for different use cases
PROMPTS = {
    "en": {
        "greeting": "Hello! How can I help you today?",
        "farewell": "Goodbye! Have a great day!",
        "error": "I'm sorry, I encountered an error. Please try again.",
        "clarification": "Could you please provide more details?",
        "content_not_found": "I couldn't find any relevant content for your question.",
        "child_safety": "I'm designed to provide safe and educational content for children."
    },
    "te": {
        "greeting": "నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?",
        "farewell": "వీడ్కోలు! మీకు మంచి రోజు కావాలని కోరుకుంటున్నాను!",
        "error": "క్షమించండి, నేను లోపాన్ని ఎదుర్కొన్నాను. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "clarification": "దయచేసి మరిన్ని వివరాలను అందించగలరా?",
        "content_not_found": "మీ ప్రశ్నకు సంబంధిత కంటెంట్‌ను నేను కనుగొనలేకపోయాను.",
        "child_safety": "నేను పిల్లలకు సురక్షితమైన మరియు విద్యాపరమైన కంటెంట్‌ను అందించడానికి రూపొందించబడ్డాను."
    },
    "ta": {
        "greeting": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "farewell": "குட்பை! நல்ல நாள் வாழ்த்துக்கள்!",
        "error": "மன்னிக்கவும், எனக்கு பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
        "clarification": "தயவுசெய்து மேலும் விவரங்களை வழங்க முடியுமா?",
        "content_not_found": "உங்கள் கேள்விக்கு தொடர்புடைய உள்ளடக்கத்தை என்னால் கண்டுபிடிக்க முடியவில்லை.",
        "child_safety": "நான் குழந்தைகளுக்கு பாதுகாப்பான மற்றும் கல்வி உள்ளடக்கத்தை வழங்க வடிவமைக்கப்பட்டுள்ளேன்."
    },
    "de": {
        "greeting": "Hallo! Wie kann ich Ihnen heute helfen?",
        "farewell": "Auf Wiedersehen! Haben Sie einen schönen Tag!",
        "error": "Es tut mir leid, ich bin auf einen Fehler gestoßen. Bitte versuchen Sie es erneut.",
        "clarification": "Könnten Sie bitte weitere Details angeben?",
        "content_not_found": "Ich konnte keine relevanten Inhalte für Ihre Frage finden.",
        "child_safety": "Ich bin darauf ausgelegt, sichere und lehrreiche Inhalte für Kinder bereitzustellen."
    },
    "hi": {
        "greeting": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूँ?",
        "farewell": "अलविदा! आपका दिन शुभ हो!",
        "error": "मुझे खेद है, मुझे एक त्रुटि मिली। कृपया पुनः प्रयास करें।",
        "clarification": "क्या आप कृपया अधिक विवरण प्रदान कर सकते हैं?",
        "content_not_found": "मुझे आपके प्रश्न के लिए कोई प्रासंगिक सामग्री नहीं मिली।",
        "child_safety": "मैं बच्चों के लिए सुरक्षित और शैक्षिक सामग्री प्रदान करने के लिए डिज़ाइन किया गया हूँ।"
    }
}

# System prompts for LLM in different languages
SYSTEM_PROMPTS = {
    "en": """You are EchoNest AI, a helpful, educational, and child-friendly assistant. 
You provide accurate, age-appropriate information and always prioritize child safety.
When you don't know something, admit it rather than making up information.
Always be respectful, patient, and encouraging.""",
    
    "te": """మీరు EchoNest AI, ఒక సహాయకరమైన, విద్యాపరమైన మరియు పిల్లలకు అనుకూలమైన సహాయకుడు.
మీరు ఖచ్చితమైన, వయసుకు తగిన సమాచారాన్ని అందిస్తారు మరియు ఎల్లప్పుడూ పిల్లల భద్రతకు ప్రాధాన్యత ఇస్తారు.
మీకు ఏదైనా తెలియకపోతే, సమాచారాన్ని తయారు చేయడం కంటే దానిని అంగీకరించండి.
ఎల్లప్పుడూ గౌరవప్రదంగా, సహనంతో మరియు ప్రోత్సాహకరంగా ఉండండి.""",
    
    "ta": """நீங்கள் EchoNest AI, ஒரு உதவிகரமான, கல்வி சார்ந்த மற்றும் குழந்தைகளுக்கு உகந்த உதவியாளர்.
நீங்கள் துல்லியமான, வயதுக்கு ஏற்ற தகவல்களை வழங்குகிறீர்கள் மற்றும் எப்போதும் குழந்தைகளின் பாதுகாப்பிற்கு முன்னுரிமை அளிக்கிறீர்கள்.
உங்களுக்கு ஏதாவது தெரியாவிட்டால், தகவல்களை உருவாக்குவதை விட அதை ஒப்புக்கொள்ளுங்கள்.
எப்போதும் மரியாதையுடனும், பொறுமையுடனும், ஊக்கமளிப்பவராகவும் இருங்கள்.""",
    
    "de": """Sie sind EchoNest AI, ein hilfreicher, lehrreicher und kinderfreundlicher Assistent.
Sie liefern genaue, altersgerechte Informationen und priorisieren stets die Sicherheit von Kindern.
Wenn Sie etwas nicht wissen, geben Sie es zu, anstatt Informationen zu erfinden.
Seien Sie immer respektvoll, geduldig und ermutigend.""",
    
    "hi": """आप EchoNest AI हैं, एक सहायक, शैक्षिक और बच्चों के अनुकूल सहायक।
आप सटीक, उम्र के अनुसार उपयुक्त जानकारी प्रदान करते हैं और हमेशा बच्चों की सुरक्षा को प्राथमिकता देते हैं।
जब आप कुछ नहीं जानते, तो जानकारी बनाने के बजाय इसे स्वीकार करें।
हमेशा सम्मानजनक, धैर्यवान और प्रोत्साहित करने वाले बनें।"""
}

# RAG prompt templates for different languages
RAG_PROMPT_TEMPLATES = {
    "en": """Answer the question based on the context below. If the question cannot be answered using the information provided, say "I don't know" instead of making up an answer.

Context: {context}

Question: {question}

Answer:""",
    
    "te": """క్రింద ఇచ్చిన సందర్భం ఆధారంగా ప్రశ్నకు సమాధానం ఇవ్వండి. అందించిన సమాచారం ఉపయోగించి ప్రశ్నకు సమాధానం ఇవ్వలేకపోతే, సమాధానాన్ని తయారు చేయడం కంటే "నాకు తెలియదు" అని చెప్పండి.

సందర్భం: {context}

ప్రశ్న: {question}

సమాధానం:""",
    
    "ta": """கீழே உள்ள சூழலின் அடிப்படையில் கேள்விக்கு பதிலளிக்கவும். வழங்கப்பட்ட தகவலைப் பயன்படுத்தி கேள்விக்கு பதிலளிக்க முடியாவிட்டால், பதிலை உருவாக்குவதற்குப் பதிலாக "எனக்குத் தெரியாது" என்று கூறுங்கள்.

சூழல்: {context}

கேள்வி: {question}

பதில்:""",
    
    "de": """Beantworten Sie die Frage basierend auf dem untenstehenden Kontext. Wenn die Frage mit den bereitgestellten Informationen nicht beantwortet werden kann, sagen Sie "Ich weiß es nicht", anstatt eine Antwort zu erfinden.

Kontext: {context}

Frage: {question}

Antwort:""",
    
    "hi": """नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर दें। यदि प्रदान की गई जानकारी का उपयोग करके प्रश्न का उत्तर नहीं दिया जा सकता है, तो उत्तर बनाने के बजाय "मुझे नहीं पता" कहें।

संदर्भ: {context}

प्रश्न: {question}

उत्तर:"""
}

def get_prompt(prompt_key: str, language: str) -> str:
    """
    Get a language-specific prompt.
    
    Args:
        prompt_key: Key for the prompt
        language: Language code
        
    Returns:
        Language-specific prompt text
    """
    # Default to English if language not supported
    if language not in PROMPTS:
        language = "en"
    
    # Default to error message if prompt key not found
    if prompt_key not in PROMPTS[language]:
        return PROMPTS[language].get("error", "An error occurred.")
    
    return PROMPTS[language][prompt_key]

def get_system_prompt(language: str) -> str:
    """
    Get the system prompt for the LLM in the specified language.
    
    Args:
        language: Language code
        
    Returns:
        System prompt for the LLM
    """
    # Default to English if language not supported
    if language not in SYSTEM_PROMPTS:
        language = "en"
    
    return SYSTEM_PROMPTS[language]

def get_rag_prompt_template(language: str) -> str:
    """
    Get the RAG prompt template for the specified language.
    
    Args:
        language: Language code
        
    Returns:
        RAG prompt template
    """
    # Default to English if language not supported
    if language not in RAG_PROMPT_TEMPLATES:
        language = "en"
    
    return RAG_PROMPT_TEMPLATES[language]

def format_rag_prompt(language: str, context: str, question: str) -> str:
    """
    Format a RAG prompt with context and question.
    
    Args:
        language: Language code
        context: Context text
        question: Question text
        
    Returns:
        Formatted RAG prompt
    """
    template = get_rag_prompt_template(language)
    return template.format(context=context, question=question)
