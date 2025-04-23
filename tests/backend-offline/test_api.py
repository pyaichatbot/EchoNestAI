import pytest
from fastapi.testclient import TestClient
import os
import tempfile
from unittest.mock import patch

def test_health_endpoint(client):
    """Test the health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_auth_login_validation(client):
    """Test login endpoint validates input correctly."""
    # Test with missing fields
    response = client.post("/api/auth/token", data={})
    assert response.status_code == 422
    
    # Test with invalid email format
    response = client.post("/api/auth/token", data={
        "username": "invalid-email",
        "password": "password123"
    })
    assert response.status_code == 422
    
    # Test with invalid credentials
    response = client.post("/api/auth/token", data={
        "username": "nonexistent@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == 401

def test_auth_register_validation(client):
    """Test registration endpoint validates input correctly."""
    # Test with missing fields
    response = client.post("/api/auth/register", json={})
    assert response.status_code == 422
    
    # Test with invalid email format
    response = client.post("/api/auth/register", json={
        "email": "invalid-email",
        "password": "Password123!",
        "firstName": "Test",
        "lastName": "User"
    })
    assert response.status_code == 422
    
    # Test with weak password
    response = client.post("/api/auth/register", json={
        "email": "test@example.com",
        "password": "weak",
        "firstName": "Test",
        "lastName": "User"
    })
    assert response.status_code == 422

def test_protected_endpoints_require_auth(client):
    """Test that protected endpoints require authentication."""
    # Test content endpoint
    response = client.get("/api/content")
    assert response.status_code == 401
    
    # Test chat endpoint
    response = client.get("/api/chat/sessions")
    assert response.status_code == 401

def test_content_operations(authenticated_client):
    """Test content operations in offline mode."""
    # Create a temporary text file for testing
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
        temp.write(b"This is test content for EchoNest AI.")
        temp_path = temp.name
    
    try:
        # Mock file upload
        with open(temp_path, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            data = {
                'title': 'Test Document',
                'description': 'A test document for offline mode',
                'language': 'en',
                'sync_offline': 'true'
            }
            
            # Upload content
            response = authenticated_client.post(
                "/api/content/upload",
                files=files,
                data=data
            )
            assert response.status_code in [200, 201]
            content_data = response.json()
            content_id = content_data.get('id')
            assert content_id is not None
            
            # Get content list
            response = authenticated_client.get("/api/content")
            assert response.status_code == 200
            contents = response.json()
            assert len(contents) > 0
            
            # Get specific content
            response = authenticated_client.get(f"/api/content/{content_id}")
            assert response.status_code == 200
            assert response.json()['title'] == 'Test Document'
            
            # Update content
            response = authenticated_client.put(
                f"/api/content/{content_id}",
                json={
                    'title': 'Updated Test Document',
                    'sync_offline': True
                }
            )
            assert response.status_code == 200
            
            # Verify update
            response = authenticated_client.get(f"/api/content/{content_id}")
            assert response.status_code == 200
            assert response.json()['title'] == 'Updated Test Document'
            
            # Delete content
            response = authenticated_client.delete(f"/api/content/{content_id}")
            assert response.status_code == 200
            
            # Verify deletion
            response = authenticated_client.get(f"/api/content/{content_id}")
            assert response.status_code == 404
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_chat_operations(authenticated_client):
    """Test chat operations in offline mode."""
    # Create chat session
    response = authenticated_client.post(
        "/api/chat/sessions",
        json={"title": "Offline Test Chat"}
    )
    assert response.status_code == 201
    session_data = response.json()
    session_id = session_data.get('id')
    assert session_id is not None
    
    # Get chat sessions
    response = authenticated_client.get("/api/chat/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) > 0
    
    # Send a message
    with patch('app.llm.chat_service.ChatService.generate_response') as mock_generate:
        # Mock the LLM response
        mock_generate.return_value = {
            "content": "This is a test response from the offline LLM.",
            "sources": []
        }
        
        response = authenticated_client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "content": "Hello, this is a test message.",
                "language": "en"
            }
        )
        assert response.status_code == 201
    
    # Get messages
    response = authenticated_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) >= 2  # User message and AI response
    
    # Delete session
    response = authenticated_client.delete(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 200
    
    # Verify deletion
    response = authenticated_client.get(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 404

def test_device_sync_endpoints(authenticated_client):
    """Test device sync endpoints."""
    # Get sync status
    response = authenticated_client.get("/api/devices/sync/status")
    assert response.status_code == 200
    
    # Test sync content endpoint
    response = authenticated_client.post(
        "/api/devices/sync/content",
        json={"device_id": "test_device_001"}
    )
    assert response.status_code in [200, 202]
    
    # Test sync manifest endpoint
    response = authenticated_client.get(
        "/api/devices/sync/manifest",
        params={"device_id": "test_device_001"}
    )
    assert response.status_code == 200
    manifest = response.json()
    assert "content" in manifest
    assert "models" in manifest
