import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models.models import ChatSession, ChatMessage
from app.services.content_processor import content_processor
from app.services.language_service import detect_language, get_language_model

logger = setup_logging("chat_service")

async def process_chat_query(
    db: AsyncSession,
    query: str,
    session_id: str,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    document_scope: Optional[List[str]] = None,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a chat query and generate a response using RAG.
    
    Args:
        db: Database session
        query: User query
        session_id: Chat session ID
        child_id: Optional child ID for context
        group_id: Optional group ID for context
        document_scope: Optional list of document IDs to search within
        language: Optional language code (auto-detected if not provided)
        
    Returns:
        Response with answer, sources, and confidence score
    """
    # Detect language if not provided
    if not language:
        language_detection = await detect_language(query)
        language = language_detection["detected_language"]
    
    # Get chat history for context
    chat_history = await get_chat_history(db, session_id, limit=5)
    
    # Search for relevant content
    search_results = await content_processor.search_content(
        db, 
        query=query, 
        language=language,
        document_scope=document_scope,
        top_k=3
    )
    
    # Extract relevant passages
    context_passages = [result["chunk_text"] for result in search_results]
    source_documents = [result["content_id"] for result in search_results]
    
    # Generate response using LLM
    response, confidence = await generate_llm_response(
        query=query,
        context_passages=context_passages,
        chat_history=chat_history,
        language=language
    )
    
    logger.info(f"Generated response for query in {language}: {query[:50]}...")
    
    return {
        "response": response,
        "source_documents": source_documents,
        "confidence": confidence,
        "used_docs": len(source_documents)
    }

async def get_chat_history(
    db: AsyncSession,
    session_id: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get chat history for a session.
    
    Args:
        db: Database session
        session_id: Chat session ID
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of chat messages with role and content
    """
    from sqlalchemy import select
    from sqlalchemy.sql import desc
    
    query = select(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(desc(ChatMessage.created_at)).limit(limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    # Convert to list of dicts and reverse to get chronological order
    history = []
    for msg in reversed(messages):
        role = "user" if msg.is_user else "assistant"
        history.append({
            "role": role,
            "content": msg.content
        })
    
    return history

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
    # Import LLM libraries
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    
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
    
    # Create prompt
    prompt = f"""
Context information:
{context}

Chat history:
{history_text}

User: {query}
Assistant: """
    
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
        return "I'm sorry, I couldn't generate a response at this time.", 0.0

async def stream_chat_response(
    query: str,
    session_id: str,
    language: str
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
    # Import LLM libraries
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
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
        yield "I'm sorry, I couldn't generate a response at this time."
    
    return tokens
