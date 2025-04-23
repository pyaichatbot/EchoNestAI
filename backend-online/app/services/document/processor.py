from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import aiofiles
import asyncio
import datetime

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.content.embeddings import generate_embeddings, store_embeddings, split_text_into_chunks

logger = setup_logging("document_processor")

async def process_document(
    content_id: str,
    file_path: str,
    language: str,
    event_manager: Any
) -> None:
    """
    Process a document file:
    1. Extract text
    2. Split into chunks
    3. Generate embeddings
    4. Store in vector database
    """
    # Update: Text extraction (20%)
    await event_manager.broadcast(
        {
            "content_id": content_id,
            "status": "processing",
            "progress": 20,
            "message": "Extracting text from document"
        },
        client_filter=content_id
    )
    
    # Extract text based on file type
    file_extension = Path(file_path).suffix.lower()
    extracted_text = ""
    
    try:
        if file_extension == ".pdf":
            extracted_text = await extract_text_from_pdf(file_path)
        elif file_extension in [".docx", ".doc"]:
            extracted_text = await extract_text_from_docx(file_path)
        elif file_extension in [".txt", ".md"]:
            extracted_text = await extract_text_from_text_file(file_path)
        else:
            # Try generic text extraction
            extracted_text = await extract_text_generic(file_path)
        
        if not extracted_text:
            raise ValueError(f"Could not extract text from {file_path}")
        
        # Update: Chunking (40%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 40,
                "message": "Splitting document into chunks"
            },
            client_filter=content_id
        )
        
        # Split text into chunks
        chunks = split_text_into_chunks(extracted_text)
        
        # Update: Embedding generation (70%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 70,
                "message": "Generating embeddings"
            },
            client_filter=content_id
        )
        
        # Generate embeddings for each chunk
        embeddings = await generate_embeddings(chunks, language)
        
        # Update: Vector storage (90%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 90,
                "message": "Storing in vector database"
            },
            client_filter=content_id
        )
        
        # Store embeddings in database
        await store_embeddings(content_id, chunks, embeddings)
        
    except Exception as e:
        logger.error(f"Error processing document {content_id}: {str(e)}")
        raise

async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        import PyPDF2
        
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

async def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        import docx
        
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise

async def extract_text_from_text_file(file_path: str) -> str:
    """Extract text from plain text file."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
            text = await file.read()
        return text
    except UnicodeDecodeError:
        # Try different encodings
        encodings = ["latin-1", "cp1252", "iso-8859-1"]
        for encoding in encodings:
            try:
                async with aiofiles.open(file_path, "r", encoding=encoding) as file:
                    text = await file.read()
                return text
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode text file with any encoding")

async def extract_text_generic(file_path: str) -> str:
    """Try to extract text using a generic method."""
    try:
        import textract
        
        # textract is not async, so run in a thread pool
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, lambda: textract.process(file_path).decode("utf-8")
        )
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text with textract: {str(e)}")
        raise
