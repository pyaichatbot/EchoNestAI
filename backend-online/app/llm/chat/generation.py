from typing import List, Dict, Any, Optional, Tuple
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.core.logging import setup_logging
from app.services.language_service import get_language_model, get_prompt, format_rag_prompt
from app.core.config import settings

logger = setup_logging("chat_generation")

async def generate_llm_response(
    query: str,
    context_passages: List[str],
    chat_history: List[Dict[str, Any]],
    language: str
) -> Tuple[str, float]:
    """
    Generate a response using a language model.
    
    Args:
        query: User query
        context_passages: Relevant passages from content search
        chat_history: Chat history for context
        language: Language code
        
    Returns:
        Generated response and confidence score
    """
    # Get appropriate model for language
    language_models = get_language_model(language)
    model_name = language_models["llm_model"]
    
    # Load model and tokenizer
    model_path = os.path.join(settings.MODELS_FOLDER, model_name)
    
    if os.path.exists(model_path):
        # Load from local path if available
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        # Download model if not available locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Save model for future use
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Prepare context from passages
    context = "\n\n".join(context_passages)
    
    # Prepare chat history
    history_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    # Get RAG prompt template for language
    prompt = format_rag_prompt(language, context, query)
    
    # Add chat history if available
    if history_text:
        prompt = f"Chat history:\n{history_text}\n\n{prompt}"
    
    # Generate response
    try:
        result = generator(prompt, return_full_text=False)[0]["generated_text"]
        
        # Calculate confidence score based on perplexity
        # Lower perplexity = higher confidence
        inputs = tokenizer(result, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        # Convert perplexity to confidence score (0-1)
        perplexity = torch.exp(outputs.loss).item()
        confidence = max(0, min(1, 1.0 - (perplexity / 100)))
        
        return result, confidence
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        
        # Get error message in appropriate language
        error_msg = get_prompt("error", language)
        
        return error_msg, 0.0

async def stream_chat_response(
    query: str,
    session_id: str,
    language: str = "en"
) -> List[str]:
    """
    Stream a chat response token by token.
    
    Args:
        query: User query
        session_id: Chat session ID
        language: Language code
        
    Returns:
        List of response tokens
    """
    # Get appropriate model for language
    language_models = get_language_model(language)
    model_name = language_models["llm_model"]
    
    # Load model and tokenizer
    model_path = os.path.join(settings.MODELS_FOLDER, model_name)
    
    if os.path.exists(model_path):
        # Load from local path if available
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        # Download model if not available locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    
    # Create simple prompt
    prompt = f"User: {query}\nAssistant: "
    
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate response token by token
    tokens = []
    
    # Set up generation parameters
    gen_kwargs = {
        "input_ids": input_ids,
        "max_length": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    try:
        # Stream tokens
        for output in model.generate(**gen_kwargs, streamer=True):
            # Decode token
            token = tokenizer.decode(output[-1], skip_special_tokens=True)
            if token:
                tokens.append(token)
                yield token
    except Exception as e:
        logger.error(f"Error streaming response: {str(e)}")
        
        # Get error message in appropriate language
        error_msg = get_prompt("error", language)
        
        yield error_msg
    
    return tokens
