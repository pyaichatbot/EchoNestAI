import pytest
from fastapi.testclient import TestClient

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
    
    # Test devices endpoint
    response = client.get("/api/devices")
    assert response.status_code == 401

def test_content_crud_operations(authenticated_client):
    """Test content CRUD operations."""
    # Get initial content list
    response = authenticated_client.get("/api/content")
    assert response.status_code == 200
    initial_count = len(response.json())
    
    # Create test content (mock file upload)
    # Note: In a real test, we would use a test file
    # This is a simplified version for demonstration
    response = authenticated_client.post(
        "/api/content/mock-upload",
        json={
            "title": "Test Content",
            "description": "Test Description",
            "language": "en",
            "type": "document",
            "sync_offline": True
        }
    )
    assert response.status_code in [201, 200]
    
    # Verify content was created
    response = authenticated_client.get("/api/content")
    assert response.status_code == 200
    assert len(response.json()) > initial_count
    
    # Get content by ID
    content_id = response.json()[0]["id"]
    response = authenticated_client.get(f"/api/content/{content_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Test Content"
    
    # Update content
    response = authenticated_client.put(
        f"/api/content/{content_id}",
        json={"title": "Updated Test Content"}
    )
    assert response.status_code == 200
    
    # Verify update
    response = authenticated_client.get(f"/api/content/{content_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Test Content"
    
    # Delete content
    response = authenticated_client.delete(f"/api/content/{content_id}")
    assert response.status_code == 200
    
    # Verify deletion
    response = authenticated_client.get(f"/api/content/{content_id}")
    assert response.status_code == 404

def test_chat_session_operations(authenticated_client):
    """Test chat session operations."""
    # Create chat session
    response = authenticated_client.post(
        "/api/chat/sessions",
        json={"title": "Test Chat Session"}
    )
    assert response.status_code == 201
    session_id = response.json()["id"]
    
    # Get session by ID
    response = authenticated_client.get(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Test Chat Session"
    
    # Send message
    response = authenticated_client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": "Hello, AI!", "language": "en"}
    )
    assert response.status_code == 201
    
    # Get messages
    response = authenticated_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert len(response.json()) >= 1  # Should have at least our message
    
    # Delete session
    response = authenticated_client.delete(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 200
    
    # Verify deletion
    response = authenticated_client.get(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 404
