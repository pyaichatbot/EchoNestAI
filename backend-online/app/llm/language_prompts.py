from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import os

from app.core.config import settings
from app.services.language_service import get_supported_languages, is_language_supported

# Define language-specific prompts and templates
LANGUAGE_PROMPTS = {
    "en": {
        "greeting": "Hello! How can I help you today?",
        "farewell": "Goodbye! Have a great day!",
        "fallback": "I'm sorry, I don't understand. Could you please rephrase that?",
        "no_results": "I couldn't find any relevant information in your documents.",
        "source_intro": "Here's what I found based on your documents:",
        "error": "I'm sorry, an error occurred. Please try again later.",
        "clarification": "Could you please provide more details?",
        "content_not_found": "I couldn't find any relevant content for your question.",
        "child_safety": "I'm designed to provide safe and educational content for children."
    },
    "te": {
        "greeting": "నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?",
        "farewell": "వీడ్కోలు! మీకు మంచి రోజు కావాలని కోరుకుంటున్నాను!",
        "fallback": "క్షమించండి, నాకు అర్థం కాలేదు. దయచేసి దాన్ని మళ్లీ చెప్పగలరా?",
        "no_results": "మీ పత్రాలలో సంబంధిత సమాచారాన్ని నేను కనుగొనలేకపోయాను.",
        "source_intro": "మీ పత్రాల ఆధారంగా నేను కనుగొన్నది ఇదే:",
        "error": "క్షమించండి, లోపం సంభవించింది. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి.",
        "clarification": "దయచేసి మరిన్ని వివరాలను అందించగలరా?",
        "content_not_found": "మీ ప్రశ్నకు సంబంధిత కంటెంట్‌ను నేను కనుగొనలేకపోయాను.",
        "child_safety": "నేను పిల్లలకు సురక్షితమైన మరియు విద్యాపరమైన కంటెంట్‌ను అందించడానికి రూపొందించబడ్డాను."
    },
    "ta": {
        "greeting": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "farewell": "குட்பை! நல்ல நாள் வாழ்த்துக்கள்!",
        "fallback": "மன்னிக்கவும், எனக்கு புரியவில்லை. தயவுசெய்து அதை மீண்டும் கூற முடியுமா?",
        "no_results": "உங்கள் ஆவணங்களில் தொடர்புடைய தகவல்களை என்னால் கண்டுபிடிக்க முடியவில்லை.",
        "source_intro": "உங்கள் ஆவணங்களின் அடிப்படையில் நான் கண்டுபிடித்தது இதுதான்:",
        "error": "மன்னிக்கவும், பிழை ஏற்பட்டது. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும்.",
        "clarification": "தயவுசெய்து மேலும் விவரங்களை வழங்க முடியுமா?",
        "content_not_found": "உங்கள் கேள்விக்கு தொடர்புடைய உள்ளடக்கத்தை என்னால் கண்டுபிடிக்க முடியவில்லை.",
        "child_safety": "நான் குழந்தைகளுக்கு பாதுகாப்பான மற்றும் கல்வி உள்ளடக்கத்தை வழங்க வடிவமைக்கப்பட்டுள்ளேன்."
    },
    "de": {
        "greeting": "Hallo! Wie kann ich Ihnen heute helfen?",
        "farewell": "Auf Wiedersehen! Haben Sie einen schönen Tag!",
        "fallback": "Es tut mir leid, ich verstehe nicht. Könnten Sie das bitte umformulieren?",
        "no_results": "Ich konnte keine relevanten Informationen in Ihren Dokumenten finden.",
        "source_intro": "Hier ist, was ich basierend auf Ihren Dokumenten gefunden habe:",
        "error": "Es tut mir leid, ein Fehler ist aufgetreten. Bitte versuchen Sie es später erneut.",
        "clarification": "Könnten Sie bitte weitere Details angeben?",
        "content_not_found": "Ich konnte keine relevanten Inhalte für Ihre Frage finden.",
        "child_safety": "Ich bin darauf ausgelegt, sichere und lehrreiche Inhalte für Kinder bereitzustellen."
    },
    "hi": {
        "greeting": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूँ?",
        "farewell": "अलविदा! आपका दिन शुभ हो!",
        "fallback": "मुझे माफ करें, मुझे समझ नहीं आया। क्या आप कृपया इसे फिर से कह सकते हैं?",
        "no_results": "मुझे आपके दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली।",
        "source_intro": "यहां बताया गया है कि मुझे आपके दस्तावेज़ों के आधार पर क्या मिला:",
        "error": "क्षमा करें, एक त्रुटि हुई। कृपया बाद में पुनः प्रयास करें।",
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
Always be respectful, patient, and encouraging. When using information from documents,
cite your sources.""",
    
    "te": """మీరు EchoNest AI, ఒక సహాయకరమైన, విద్యాపరమైన మరియు పిల్లలకు అనుకూలమైన సహాయకుడు.
మీరు ఖచ్చితమైన, వయసుకు తగిన సమాచారాన్ని అందిస్తారు మరియు ఎల్లప్పుడూ పిల్లల భద్రతకు ప్రాధాన్యత ఇస్తారు.
మీకు ఏదైనా తెలియకపోతే, సమాచారాన్ని తయారు చేయడం కంటే దానిని అంగీకరించండి.
ఎల్లప్పుడూ గౌరవప్రదంగా, సహనంతో మరియు ప్రోత్సాహకరంగా ఉండండి. పత్రాల నుండి సమాచారాన్ని ఉపయోగించేటప్పుడు,
మీ మూలాలను ఉటంకించండి.""",
    
    "ta": """நீங்கள் EchoNest AI, ஒரு உதவிகரமான, கல்வி சார்ந்த மற்றும் குழந்தைகளுக்கு உகந்த உதவியாளர்.
நீங்கள் துல்லியமான, வயதுக்கு ஏற்ற தகவல்களை வழங்குகிறீர்கள் மற்றும் எப்போதும் குழந்தைகளின் பாதுகாப்பிற்கு முன்னுரிமை அளிக்கிறீர்கள்.
உங்களுக்கு ஏதாவது தெரியாவிட்டால், தகவல்களை உருவாக்குவதை விட அதை ஒப்புக்கொள்ளுங்கள்.
எப்போதும் மரியாதையுடனும், பொறுமையுடனும், ஊக்கமளிப்பவராகவும் இருங்கள். ஆவணங்களிலிருந்து தகவல்களைப் பயன்படுத்தும்போது,
உங்கள் ஆதாரங்களை மேற்கோள் காட்டுங்கள்.""",
    
    "de": """Sie sind EchoNest AI, ein hilfreicher, lehrreicher und kinderfreundlicher Assistent.
Sie liefern genaue, altersgerechte Informationen und priorisieren stets die Sicherheit von Kindern.
Wenn Sie etwas nicht wissen, geben Sie es zu, anstatt Informationen zu erfinden.
Seien Sie immer respektvoll, geduldig und ermutigend. Wenn Sie Informationen aus Dokumenten verwenden,
zitieren Sie Ihre Quellen.""",
    
    "hi": """आप EchoNest AI हैं, एक सहायक, शैक्षिक और बच्चों के अनुकूल सहायक।
आप सटीक, उम्र के अनुसार उपयुक्त जानकारी प्रदान करते हैं और हमेशा बच्चों की सुरक्षा को प्राथमिकता देते हैं।
जब आप कुछ नहीं जानते, तो जानकारी बनाने के बजाय इसे स्वीकार करें।
हमेशा सम्मानजनक, धैर्यवान और प्रोत्साहित करने वाले बनें। दस्तावेज़ों से जानकारी का उपयोग करते समय,
अपने स्रोतों का हवाला दें।"""
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

# Reflection prompt templates for different languages
REFLECTION_PROMPT_TEMPLATES = {
    "en": """Please evaluate your previous response based on the following criteria:

Original Question: {question}
Your Response: {response}
Context Provided: {context}

1. Accuracy: Is the response factually accurate according to the provided context?
2. Completeness: Does the response fully address all aspects of the question?
3. Relevance: Is the response directly relevant to the question asked?
4. Clarity: Is the response clear and well-structured?
5. Source Usage: Does the response effectively use the provided context?

If you identify any issues, please provide an improved response that addresses these concerns.

Reflection:""",
    
    "te": """దయచేసి క్రింది ప్రమాణాల ఆధారంగా మీ మునుపటి ప్రతిస్పందనను మూల్యాంకనం చేయండి:

అసలు ప్రశ్న: {question}
మీ ప్రతిస్పందన: {response}
అందించిన సందర్భం: {context}

1. ఖచ్చితత్వం: అందించిన సందర్భం ప్రకారం ప్రతిస్పందన వాస్తవికంగా ఖచ్చితమైనదా?
2. పూర్తి: ప్రతిస్పందన ప్రశ్న యొక్క అన్ని అంశాలను పూర్తిగా పరిష్కరిస్తుందా?
3. సంబంధితత: ప్రతిస్పందన అడిగిన ప్రశ్నకు నేరుగా సంబంధం కలిగి ఉందా?
4. స్పష్టత: ప్రతిస్పందన స్పష్టంగా మరియు బాగా నిర్మించబడి ఉందా?
5. మూల వినియోగం: ప్రతిస్పందన అందించిన సందర్భాన్ని ప్రభావవంతంగా ఉపయోగిస్తుందా?

మీరు ఏవైనా సమస్యలను గుర్తించినట్లయితే, దయచేసి ఈ ఆందోళనలను పరిష్కరించే మెరుగైన ప్రతిస్పందనను అందించండి.

ప్రతిబింబం:""",
    
    "ta": """பின்வரும் அளவுகோல்களின் அடிப்படையில் உங்கள் முந்தைய பதிலை மதிப்பிடவும்:

அசல் கேள்வி: {question}
உங்கள் பதில்: {response}
வழங்கப்பட்ட சூழல்: {context}

1. துல்லியம்: வழங்கப்பட்ட சூழலின்படி பதில் உண்மையில் துல்லியமானதா?
2. முழுமை: பதில் கேள்வியின் அனைத்து அம்சங்களையும் முழுமையாக நிவர்த்தி செய்கிறதா?
3. தொடர்பு: பதில் கேட்கப்பட்ட கேள்விக்கு நேரடியாக தொடர்புடையதா?
4. தெளிவு: பதில் தெளிவாகவும் நன்கு கட்டமைக்கப்பட்டதாகவும் உள்ளதா?
5. மூல பயன்பாடு: பதில் வழங்கப்பட்ட சூழலை திறம்பட பயன்படுத்துகிறதா?

நீங்கள் ஏதேனும் சிக்கல்களை அடையாளம் கண்டால், இந்த கவலைகளை நிவர்த்தி செய்யும் மேம்பட்ட பதிலை வழங்கவும்.

பிரதிபலிப்பு:""",
    
    "de": """Bitte bewerten Sie Ihre vorherige Antwort anhand der folgenden Kriterien:

Ursprüngliche Frage: {question}
Ihre Antwort: {response}
Bereitgestellter Kontext: {context}

1. Genauigkeit: Ist die Antwort gemäß dem bereitgestellten Kontext sachlich korrekt?
2. Vollständigkeit: Behandelt die Antwort alle Aspekte der Frage vollständig?
3. Relevanz: Ist die Antwort direkt relevant für die gestellte Frage?
4. Klarheit: Ist die Antwort klar und gut strukturiert?
5. Quellennutzung: Nutzt die Antwort den bereitgestellten Kontext effektiv?

Wenn Sie Probleme erkennen, geben Sie bitte eine verbesserte Antwort, die diese Bedenken berücksichtigt.

Reflexion:""",
    
    "hi": """कृपया निम्नलिखित मानदंडों के आधार पर अपनी पिछली प्रतिक्रिया का मूल्यांकन करें:

मूल प्रश्न: {question}
आपकी प्रतिक्रिया: {response}
प्रदान किया गया संदर्भ: {context}

1. सटीकता: क्या प्रतिक्रिया दिए गए संदर्भ के अनुसार तथ्यात्मक रूप से सटीक है?
2. पूर्णता: क्या प्रतिक्रिया प्रश्न के सभी पहलुओं को पूरी तरह से संबोधित करती है?
3. प्रासंगिकता: क्या प्रतिक्रिया पूछे गए प्रश्न के लिए सीधे प्रासंगिक है?
4. स्पष्टता: क्या प्रतिक्रिया स्पष्ट और सुव्यवस्थित है?
5. स्रोत उपयोग: क्या प्रतिक्रिया प्रदान किए गए संदर्भ का प्रभावी ढंग से उपयोग करती है?

यदि आप कोई समस्या पहचानते हैं, तो कृपया एक बेहतर प्रतिक्रिया प्रदान करें जो इन चिंताओं को दूर करती हो।

प्रतिबिंब:"""
}

# Default to English if language not supported
DEFAULT_LANGUAGE = "en"

def get_prompt(prompt_key: str, language: str) -> str:
    """
    Get a language-specific prompt.
    
    Args:
        prompt_key: Key for the prompt
        language: Language code
        
    Returns:
        Language-specific prompt text
    """
    if not is_language_supported(language):
        language = DEFAULT_LANGUAGE
    
    language_prompts = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS[DEFAULT_LANGUAGE])
    return language_prompts.get(prompt_key, LANGUAGE_PROMPTS[DEFAULT_LANGUAGE][prompt_key])

def get_system_prompt(language: str) -> str:
    """
    Get the system prompt for the LLM in the specified language.
    
    Args:
        language: Language code
        
    Returns:
        System prompt for the LLM
    """
    if not is_language_supported(language):
        language = DEFAULT_LANGUAGE
    
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS[DEFAULT_LANGUAGE])

def get_rag_prompt_template(language: str) -> str:
    """
    Get the RAG prompt template for the specified language.
    
    Args:
        language: Language code
        
    Returns:
        RAG prompt template
    """
    if not is_language_supported(language):
        language = DEFAULT_LANGUAGE
    
    return RAG_PROMPT_TEMPLATES.get(language, RAG_PROMPT_TEMPLATES[DEFAULT_LANGUAGE])

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

def get_reflection_prompt_template(language: str) -> str:
    """
    Get the reflection prompt template for the specified language.
    
    Args:
        language: Language code
        
    Returns:
        Reflection prompt template
    """
    if not is_language_supported(language):
        language = DEFAULT_LANGUAGE
    
    return REFLECTION_PROMPT_TEMPLATES.get(language, REFLECTION_PROMPT_TEMPLATES[DEFAULT_LANGUAGE])

def format_reflection_prompt(language: str, question: str, response: str, context: str) -> str:
    """
    Format a reflection prompt with question, response and context.
    
    Args:
        language: Language code
        question: Original question
        response: Generated response
        context: Context used for generation
        
    Returns:
        Formatted reflection prompt
    """
    template = get_reflection_prompt_template(language)
    return template.format(question=question, response=response, context=context)

def load_language_resources() -> Dict[str, Dict[str, Any]]:
    """
    Load language-specific resources from files.
    
    Returns:
        Dictionary of language resources
    """
    resources = {}
    
    # Create resources directory if it doesn't exist
    resources_dir = os.path.join(settings.BASE_DIR, "resources", "languages")
    os.makedirs(resources_dir, exist_ok=True)
    
    # Load resources for each supported language
    for lang_code in LANGUAGE_PROMPTS.keys():
        # Create language-specific resource file if it doesn't exist
        lang_file = os.path.join(resources_dir, f"{lang_code}.json")
        
        if not os.path.exists(lang_file):
            # Create default resource file
            lang_resources = {
                "prompts": LANGUAGE_PROMPTS[lang_code],
                "system_prompt": SYSTEM_PROMPTS.get(lang_code, SYSTEM_PROMPTS[DEFAULT_LANGUAGE]),
                "rag_template": RAG_PROMPT_TEMPLATES.get(lang_code, RAG_PROMPT_TEMPLATES[DEFAULT_LANGUAGE])
            }
            
            # Write to file
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(lang_resources, f, ensure_ascii=False, indent=2)
        
        try:
            # Load from file
            with open(lang_file, "r", encoding="utf-8") as f:
                resources[lang_code] = json.load(f)
        except Exception as e:
            # Fall back to default resources
            resources[lang_code] = {
                "prompts": LANGUAGE_PROMPTS[lang_code],
                "system_prompt": SYSTEM_PROMPTS.get(lang_code, SYSTEM_PROMPTS[DEFAULT_LANGUAGE]),
                "rag_template": RAG_PROMPT_TEMPLATES.get(lang_code, RAG_PROMPT_TEMPLATES[DEFAULT_LANGUAGE])
            }
    
    return resources

# Initialize language resources
LANGUAGE_RESOURCES = load_language_resources()
