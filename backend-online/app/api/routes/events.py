from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
import asyncio

from app.db.database import get_db
from app.api.deps.auth import get_current_active_user
from app.sse.event_manager import system_events_manager

router = APIRouter(tags=["events"])

@router.get("/events/stream")
async def events_stream(
    request: Request,
    current_user: Any = Depends(get_current_active_user)
) -> Any:
    """
    Stream system events using Server-Sent Events.
    """
    async def event_generator():
        client_id = f"user_{current_user.id}"
        queue = system_events_manager.register_client(client_id)
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                message = await queue.get()
                if message == "CLOSE":
                    break
                
                yield {
                    "event": "message",
                    "data": message
                }
                
                await asyncio.sleep(0.1)
        finally:
            system_events_manager.remove_client(client_id)
    
    return EventSourceResponse(event_generator())
