from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.db.schemas.chat import ChatRequest, ChatResponse
from app.llm.language_prompts import format_rag_prompt, format_reflection_prompt, get_prompt
from app.services.language_service import get_language_model, is_language_supported
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models.models import Content, ContentEmbedding
from app.db.schemas.content import RelevantDocument
from app.services.document_retrieval import get_document_retriever

logger = setup_logging("chat_service")

# Cache for loaded models to prevent reloading
_model_cache = {}

async def process_chat_query(
    db: AsyncSession,
    query: str,
    session_id: str,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    document_scope: Optional[List[str]] = None,
    language: str = "en",
    context: Optional[str] = None
) -> ChatResponse:
    """
    Process a chat query using RAG (Retrieval Augmented Generation).
    If context is provided, use it as the only context_passage for the LLM; otherwise, use retrieved documents.
    
    Args:
        db: Database session
        query: User query
        session_id: Chat session ID
        child_id: Optional child ID for filtering
        group_id: Optional group ID for filtering
        document_scope: Optional list of document IDs to search within
        language: Language code
        context: Optional context string provided by frontend
        
    Returns:
        ChatResponse object containing the generated response and metadata
    """
    # Validate language
    if not is_language_supported(language):
        language = "en"  # Default to English if unsupported
    
    # Get language models
    language_models = get_language_model(language)
    
    # Log the query and reflection settings
    logger.info(f"Processing chat query in {language}: {query}")
    logger.info(f"Using models: {language_models}")
    logger.info(f"Reflection enabled: {settings.ENABLE_REFLECTION}, threshold: {settings.REFLECTION_THRESHOLD}")
    
    # Get chat history for context
    chat_history = await get_chat_history(db, session_id, limit=5)
    
    # If context is provided by frontend, use it as the only context_passage
    if context:
        context_passages = [context]
        source_documents = []
    else:
        # Generate embeddings for the query
        query_embedding = await generate_embedding(query, language)
        # Get document retriever
        retriever = await get_document_retriever()
        # Retrieve relevant documents
        documents, total = await retriever.retrieve_documents(
            db=db,
            query=query,
            query_embedding=query_embedding,
            language=language,
            document_ids=document_scope,
            user_id=child_id,
            group_ids=[group_id] if group_id else None,
            limit=3,
            min_score=0.6
        )
        context_passages = [doc["content"] for doc in documents]
        source_documents = [doc["id"] for doc in documents]
    
    # Generate response with context using reflection settings from config
    response, confidence = await generate_llm_response(
        query=query,
        context_passages=context_passages,
        chat_history=chat_history,
        language=language,
        reflection_threshold=settings.REFLECTION_THRESHOLD,
        enable_reflection=settings.ENABLE_REFLECTION
    )
    
    return ChatResponse(
        response=response,
        source_documents=source_documents,
        confidence=confidence,
        used_docs=len(source_documents)
    )

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
    from app.db.models.models import ChatMessage
    
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

async def generate_embedding(text: str, language: str) -> np.ndarray:
    """
    Generate embedding for text.
    
    Args:
        text: Text to embed
        language: Language code
        
    Returns:
        Embedding vector
    """
    # Get appropriate model for language
    language_models = get_language_model(language)
    model_name = language_models["embedding_model"]
    
    # Import embedding model
    from sentence_transformers import SentenceTransformer
    
    # Load model
    model_path = os.path.join(settings.MODELS_FOLDER, model_name)
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer(model_name)
        # Save model for future use
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        model.save(model_path)
    
    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

async def retrieve_relevant_documents(
    db: AsyncSession,
    query_embedding: np.ndarray,
    document_scope: Optional[List[str]] = None,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    top_k: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Retrieve relevant documents based on query embedding.
    
    Args:
        db: Database session
        query_embedding: Query embedding vector
        document_scope: Optional list of document IDs to search within
        child_id: Optional child ID for filtering
        group_id: Optional group ID for filtering
        top_k: Number of top results to return
        
    Returns:
        Tuple of (context passages, source document IDs)
    """
    from sqlalchemy import select
    from app.db.models.models import ContentEmbedding, Content, ContentAssignment
    
    # Build query for content embeddings
    query = select(ContentEmbedding)
    
    # Apply document scope filter if provided
    if document_scope:
        query = query.filter(ContentEmbedding.content_id.in_(document_scope))
    
    # Apply child/group filters if provided
    if child_id or group_id:
        query = query.join(Content, Content.id == ContentEmbedding.content_id)
        query = query.join(ContentAssignment, ContentAssignment.content_id == Content.id)
        
        if child_id:
            query = query.filter(ContentAssignment.child_id == child_id)
        if group_id:
            query = query.filter(ContentAssignment.group_id == group_id)
    
    # Execute query
    result = await db.execute(query)
    embeddings = result.scalars().all()
    
    # Calculate similarity scores
    search_results = []
    for emb in embeddings:
        # Parse embedding vector from JSON
        vector = np.array(json.loads(emb.embedding_vector))
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding, vector)
        
        search_results.append({
            'content_id': emb.content_id,
            'chunk_index': emb.chunk_index,
            'chunk_text': emb.chunk_text,
            'similarity': float(similarity)
        })
    
    # Sort by similarity (descending) and take top_k
    search_results.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = search_results[:top_k]
    
    # Extract context passages and source documents
    context_passages = [result['chunk_text'] for result in top_results]
    source_documents = list(set([result['content_id'] for result in top_results]))
    
    return context_passages, source_documents

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and cache model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    
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
    
    _model_cache[model_name] = (model, tokenizer)
    return model, tokenizer

async def generate_llm_response(
    query: str,
    context_passages: List[str],
    chat_history: List[Dict[str, Any]],
    language: str,
    reflection_threshold: float = settings.REFLECTION_THRESHOLD,
    enable_reflection: bool = settings.ENABLE_REFLECTION
) -> Tuple[str, float]:
    """
    Generate a response using a language model with optional reflection.
    
    Args:
        query: User query
        context_passages: Relevant passages from content search
        chat_history: Chat history for context
        language: Language code
        reflection_threshold: Confidence threshold below which reflection is triggered
        enable_reflection: Whether to enable reflection mechanism
        
    Returns:
        Generated response and confidence score
    """
    # Get appropriate model for language
    language_models = get_language_model(language)
    model_name = language_models["llm_model"]
    
    # Load model and tokenizer
    model, tokenizer = await load_model_and_tokenizer(model_name)
    
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
    
    # Generate initial response
    try:
        initial_result = generator(prompt, return_full_text=False)[0]["generated_text"]
        
        # Calculate confidence score based on perplexity
        inputs = tokenizer(initial_result, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        # Convert perplexity to confidence score (0-1)
        perplexity = torch.exp(outputs.loss).item()
        confidence = max(0, min(1, 1.0 - (perplexity / 100)))
        
        # Check if reflection is needed
        if enable_reflection and confidence < reflection_threshold:
            logger.info(f"Low confidence ({confidence:.2f}), triggering reflection")
            
            # Get reflection prompt
            reflection_prompt = format_reflection_prompt(
                language=language,
                question=query,
                response=initial_result,
                context=context
            )
            
            # Generate reflection
            reflection_result = generator(reflection_prompt, return_full_text=False)[0]["generated_text"]
            
            # If reflection suggests improvements, use the improved response
            if "improved response" in reflection_result.lower():
                # Extract improved response from reflection
                improved_response = reflection_result.split("improved response:", 1)[-1].strip()
                if improved_response:
                    logger.info("Using improved response from reflection")
                    return improved_response, confidence
            
            # Log reflection for analysis
            logger.info(f"Reflection output: {reflection_result}")
        
        return initial_result, confidence
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        error_msg = get_prompt("error", language)
        return error_msg, 0.0

async def stream_chat_response(
    db: AsyncSession,
    query: str,
    session_id: str,
    language: str = "en",
    enable_reflection: bool = settings.ENABLE_REFLECTION,
    reflection_threshold: float = settings.REFLECTION_THRESHOLD,
    document_scope: Optional[List[str]] = None,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Stream chat response with RAG"""
    try:
        # Get chat history
        chat_history = await get_chat_history(db, session_id, limit=5)
        
        # Generate query embedding
        query_embedding = await generate_embedding(query, language)
        
        # Get document retriever
        retriever = await get_document_retriever()
        
        # Retrieve relevant documents
        documents, _ = await retriever.retrieve_documents(
            db=db,
            query=query,
            query_embedding=query_embedding,
            language=language,
            document_ids=document_scope,
            user_id=child_id,
            group_ids=[group_id] if group_id else None,
            limit=3,
            min_score=0.6
        )
        
        # Extract context passages
        context_passages = [doc["content"] for doc in documents]
        
        # Get language models
        language_models = get_language_model(language)
        model_name = language_models["llm_model"]
        
        # Load model and tokenizer
        model, tokenizer = await load_model_and_tokenizer(model_name)
        
        # Create streaming pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Prepare context and prompt
        context = "\n\n".join(context_passages) if context_passages else ""
        prompt = format_rag_prompt(language, context, query)
        
        # Add chat history if available
        history_text = ""
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        if history_text:
            prompt = f"Chat history:\n{history_text}\n\n{prompt}"
        
        # Generate response tokens
        for output in generator(prompt, return_full_text=False, streaming=True):
            yield output[0]["generated_text"]
            
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}", exc_info=True)
        yield f"Error: {str(e)}"

async def get_relevant_documents(
    db: AsyncSession,
    query: str,
    language: str = "en",
    document_ids: Optional[List[str]] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.6
) -> List[RelevantDocument]:
    """Get relevant documents for a query"""
    try:
        # Generate query embedding
        query_embedding = await generate_embedding(query, language)
        
        # Get document retriever
        retriever = await get_document_retriever()
        
        # Retrieve documents
        documents, _ = await retriever.retrieve_documents(
            db=db,
            query=query,
            query_embedding=query_embedding,
            language=language,
            document_ids=document_ids,
            limit=top_k,
            min_score=similarity_threshold
        )
        
        # Convert to RelevantDocument schema
        relevant_docs = []
        for doc in documents:
            relevant_doc = RelevantDocument(
                id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                relevance_score=doc["relevance_score"],
                language=doc["language"],
                type=doc["type"],
                metadata=doc["metadata"]
            )
            relevant_docs.append(relevant_doc)
        
        return relevant_docs
        
    except Exception as e:
        logger.error(f"Error getting relevant documents: {str(e)}", exc_info=True)
        return []
