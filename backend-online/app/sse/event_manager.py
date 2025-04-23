import asyncio
from typing import Dict, Any, Optional
import json

class EventManager:
    """
    Manager for Server-Sent Events (SSE) clients.
    Handles registration, message broadcasting, and client removal.
    """
    
    def __init__(self):
        self.clients: Dict[str, asyncio.Queue] = {}
    
    def register_client(self, client_id: str) -> asyncio.Queue:
        """
        Register a new client and return its message queue.
        """
        if client_id in self.clients:
            return self.clients[client_id]
        
        self.clients[client_id] = asyncio.Queue()
        return self.clients[client_id]
    
    def remove_client(self, client_id: str) -> None:
        """
        Remove a client from the manager.
        """
        if client_id in self.clients:
            del self.clients[client_id]
    
    async def broadcast(self, message: Any, client_filter: Optional[str] = None) -> None:
        """
        Broadcast a message to all clients or filtered clients.
        """
        if not self.clients:
            return
        
        # Convert message to JSON string if it's not already a string
        if not isinstance(message, str):
            message = json.dumps(message)
        
        # Send to all clients or filtered clients
        for client_id, queue in self.clients.items():
            if client_filter is None or client_filter in client_id:
                await queue.put(message)
    
    async def send_to_client(self, client_id: str, message: Any) -> bool:
        """
        Send a message to a specific client.
        Returns True if the client exists and the message was sent.
        """
        if client_id not in self.clients:
            return False
        
        # Convert message to JSON string if it's not already a string
        if not isinstance(message, str):
            message = json.dumps(message)
        
        await self.clients[client_id].put(message)
        return True
    
    async def close_client(self, client_id: str) -> None:
        """
        Send a close message to a client and remove it.
        """
        if client_id in self.clients:
            await self.clients[client_id].put("CLOSE")
            self.remove_client(client_id)

# Create global event managers
chat_stream_manager = EventManager()
content_upload_manager = EventManager()
system_events_manager = EventManager()
