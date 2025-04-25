from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Union, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError
import traceback
import json

from app.core.logging import setup_logging

logger = setup_logging("error_handler")

class ErrorHandler:
    async def __call__(self, request: Request, call_next):
        try:
            return await call_next(request)
            
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": json.loads(e.json()),
                    "message": "Validation error"
                }
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": "Database error occurred"
                }
            )
            
        except HTTPException as e:
            # Re-raise HTTP exceptions (they're handled by FastAPI)
            raise
            
        except Exception as e:
            # Log unexpected errors
            logger.error(
                f"Unexpected error: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": "An unexpected error occurred"
                }
            )

class RequestValidator:
    async def __call__(self, request: Request, call_next):
        # Validate content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "message": "Request too large"
                }
            )
        
        # Validate content type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith(("application/json", "multipart/form-data")):
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={
                        "message": "Unsupported media type"
                    }
                )
        
        return await call_next(request)

class RequestLogger:
    async def __call__(self, request: Request, call_next):
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Client: {request.client.host}"
        )
        
        # Process request and capture response
        response = await call_next(request)
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"Path: {request.url.path}"
        )
        
        return response 