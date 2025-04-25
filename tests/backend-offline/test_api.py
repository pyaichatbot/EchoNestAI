import pytest
from fastapi.testclient import TestClient
import os
import tempfile
from unittest.mock import patch
import uuid

def test_root_endpoint(client: TestClient):
    """Test the root endpoint returns the expected status message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "EchoNest AI Offline Backend is running"}

def test_health_endpoint(client: TestClient):
    """Test the health endpoint returns 200 OK and expected status/version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # Add assertion for version and environment if available in your config
    # assert "version" in data
    # assert "environment" in data

def test_auth_register(client: TestClient):
    """Test user registration with valid data."""
    user_data = {
        "email": "test@example.com",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    }
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["email"] == user_data["email"]
    assert data["firstName"] == user_data["firstName"]
    assert data["lastName"] == user_data["lastName"]
    assert "hashed_password" not in data  # Password should not be returned

def test_auth_register_duplicate_email(client: TestClient):
    """Test registration with existing email."""
    # First register a user
    user_data = {
        "email": "duplicate@example.com",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    }
    client.post("/api/v1/auth/register", json=user_data)
    
    # Try to register again with the same email
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"

def test_auth_register_validation(client: TestClient):
    """Test input validation for registration."""
    # Test with missing fields
    response = client.post("/api/v1/auth/register", json={})
    assert response.status_code == 422
    
    # Test with invalid email format
    response = client.post("/api/v1/auth/register", json={
        "email": "invalid-email",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    })
    assert response.status_code == 422
    
    # Test with weak password
    response = client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "weak",
        "firstName": "Test",
        "lastName": "User"
    })
    assert response.status_code == 422

def test_auth_login(client: TestClient):
    """Test successful login and token generation."""
    # First register a user
    user_data = {
        "email": "login@example.com",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    }
    client.post("/api/v1/auth/register", json=user_data)
    
    # Test login
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": user_data["email"],
            "password": user_data["password"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "expires_in" in data
    assert "role" in data

def test_auth_login_invalid_credentials(client: TestClient):
    """Test login with invalid credentials."""
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "nonexistent@example.com",
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"

def test_auth_get_current_user(authenticated_client: TestClient):
    """Test retrieving current user info with valid token."""
    response = authenticated_client.get("/api/v1/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "email" in data
    assert "firstName" in data
    assert "lastName" in data
    assert "hashed_password" not in data

def test_auth_update_current_user(authenticated_client: TestClient):
    """Test updating current user info."""
    update_data = {
        "firstName": "Updated",
        "lastName": "Name"
    }
    response = authenticated_client.put("/api/v1/auth/me", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["firstName"] == update_data["firstName"]
    assert data["lastName"] == update_data["lastName"]

def test_auth_forgot_password(client: TestClient):
    """Test password reset request."""
    # First register a user
    user_data = {
        "email": "reset@example.com",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    }
    client.post("/api/v1/auth/register", json=user_data)
    
    # Test forgot password
    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": user_data["email"]}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "If the email exists, a password reset link has been sent"

def test_auth_reset_password(client: TestClient):
    """Test password reset with valid token."""
    # This test requires a valid reset token from the database
    # In a real test, we would need to get the token from the database
    # For now, we'll skip this test
    pytest.skip("Requires valid reset token from database")

def test_auth_reset_password_invalid_token(client: TestClient):
    """Test password reset with invalid token."""
    response = client.post(
        "/api/v1/auth/reset-password",
        json={
            "token": "invalid-token",
            "new_password": "NewPassword123!"
        }
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired token"

def test_auth_assign_role(admin_client: TestClient):
    """Test role assignment by admin."""
    # First create a regular user
    user_data = {
        "email": "role@example.com",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    }
    response = client.post("/api/v1/auth/register", json=user_data)
    user_id = response.json()["id"]
    
    # Test role assignment
    response = admin_client.post(
        "/api/v1/auth/assign-role",
        json={
            "user_id": user_id,
            "role": "admin"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "admin"

def test_auth_assign_role_unauthorized(authenticated_client: TestClient):
    """Test role assignment without admin privileges."""
    response = authenticated_client.post(
        "/api/v1/auth/assign-role",
        json={
            "user_id": "some-user-id",
            "role": "admin"
        }
    )
    assert response.status_code == 403  # Forbidden

def test_protected_endpoints_require_auth(client: TestClient):
    """Test that protected endpoints require authentication."""
    endpoints = [
        "/api/v1/content",
        "/api/v1/chat/sessions",
        "/api/v1/devices",
        "/api/v1/auth/me"
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 401

def test_invalid_token_access(client: TestClient):
    """Test accessing protected routes with invalid tokens."""
    headers = {"Authorization": "Bearer invalid-token"}
    endpoints = [
        "/api/v1/content",
        "/api/v1/chat/sessions",
        "/api/v1/devices",
        "/api/v1/auth/me"
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint, headers=headers)
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

# Content Management Tests

def test_content_list(authenticated_client: TestClient):
    """Test listing contents with optional filters."""
    response = authenticated_client.get("/api/v1/content")
    assert response.status_code == 200
    contents = response.json()
    assert isinstance(contents, list)
    
    # Test with filters
    response = authenticated_client.get(
        "/api/v1/content",
        params={
            "content_type": "document",
            "language": "en",
            "skip": 0,
            "limit": 10
        }
    )
    assert response.status_code == 200
    filtered_contents = response.json()
    assert isinstance(filtered_contents, list)

def test_content_upload(authenticated_client: TestClient):
    """Test uploading content with file."""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"Test content for upload")
        temp_file_path = temp_file.name
    
    try:
        # Upload the file
        with open(temp_file_path, "rb") as f:
            response = authenticated_client.post(
                "/api/v1/content",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "title": "Test Content",
                    "content_type": "document",
                    "language": "en",
                    "sync_offline": "true",
                    "assign_to": "[]"
                }
            )
        assert response.status_code == 200
        content = response.json()
        assert "id" in content
        assert content["title"] == "Test Content"
        assert content["type"] == "document"
        assert content["language"] == "en"
        assert content["sync_offline"] is True
    
    finally:
        # Clean up
        os.unlink(temp_file_path)

def test_content_upload_validation(authenticated_client: TestClient):
    """Test content upload validation."""
    # Test with missing required fields
    response = authenticated_client.post(
        "/api/v1/content",
        files={"file": ("test.txt", b"test", "text/plain")},
        data={}
    )
    assert response.status_code == 422
    
    # Test with invalid content type
    response = authenticated_client.post(
        "/api/v1/content",
        files={"file": ("test.txt", b"test", "text/plain")},
        data={
            "title": "Test",
            "content_type": "invalid",
            "language": "en"
        }
    )
    assert response.status_code == 400
    assert "Invalid content type" in response.json()["detail"]

def test_content_get(authenticated_client: TestClient):
    """Test getting content by ID."""
    # First create a content item
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Test content")
        response = authenticated_client.post(
            "/api/v1/content",
            files={"file": ("test.txt", temp_file, "text/plain")},
            data={
                "title": "Test Get Content",
                "content_type": "document",
                "language": "en"
            }
        )
    content_id = response.json()["id"]
    
    # Test getting the content
    response = authenticated_client.get(f"/api/v1/content/{content_id}")
    assert response.status_code == 200
    content = response.json()
    assert content["id"] == content_id
    assert content["title"] == "Test Get Content"

def test_content_get_not_found(authenticated_client: TestClient):
    """Test getting non-existent content."""
    response = authenticated_client.get("/api/v1/content/nonexistent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Content not found"

def test_content_update(authenticated_client: TestClient):
    """Test updating content."""
    # First create a content item
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Test content")
        response = authenticated_client.post(
            "/api/v1/content",
            files={"file": ("test.txt", temp_file, "text/plain")},
            data={
                "title": "Original Title",
                "content_type": "document",
                "language": "en"
            }
        )
    content_id = response.json()["id"]
    
    # Test updating the content
    update_data = {
        "title": "Updated Title",
        "language": "es"
    }
    response = authenticated_client.put(
        f"/api/v1/content/{content_id}",
        json=update_data
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == update_data["title"]
    assert content["language"] == update_data["language"]

def test_content_update_unauthorized(authenticated_client: TestClient):
    """Test updating content without proper permissions."""
    # This would require creating content with a different user
    # For now, we'll skip this test
    pytest.skip("Requires content created by different user")

def test_content_delete(authenticated_client: TestClient):
    """Test deleting content."""
    # First create a content item
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Test content")
        response = authenticated_client.post(
            "/api/v1/content",
            files={"file": ("test.txt", temp_file, "text/plain")},
            data={
                "title": "Content to Delete",
                "content_type": "document",
                "language": "en"
            }
        )
    content_id = response.json()["id"]
    
    # Test deleting the content
    response = authenticated_client.delete(f"/api/v1/content/{content_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Content deleted successfully"
    
    # Verify content is deleted
    response = authenticated_client.get(f"/api/v1/content/{content_id}")
    assert response.status_code == 404

def test_content_assign(authenticated_client: TestClient):
    """Test assigning content to a target."""
    # First create a content item
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Test content")
        response = authenticated_client.post(
            "/api/v1/content",
            files={"file": ("test.txt", temp_file, "text/plain")},
            data={
                "title": "Content to Assign",
                "content_type": "document",
                "language": "en"
            }
        )
    content_id = response.json()["id"]
    
    # Test assigning the content
    assignment_data = {
        "content_id": content_id,
        "target_id": "test-target-id",
        "target_type": "child"
    }
    response = authenticated_client.post(
        "/api/v1/content/assign",
        json=assignment_data
    )
    assert response.status_code == 200
    assignment = response.json()
    assert assignment["content_id"] == content_id
    assert assignment["target_id"] == assignment_data["target_id"]

def test_content_search(authenticated_client: TestClient):
    """Test searching content."""
    response = authenticated_client.get(
        "/api/v1/content/search",
        params={
            "query": "test",
            "language": "en",
            "top_k": 5
        }
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)

def test_content_reprocess(admin_client: TestClient):
    """Test reprocessing content (admin only)."""
    # First create a content item
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Test content")
        response = admin_client.post(
            "/api/v1/content",
            files={"file": ("test.txt", temp_file, "text/plain")},
            data={
                "title": "Content to Reprocess",
                "content_type": "document",
                "language": "en"
            }
        )
    content_id = response.json()["id"]
    
    # Test reprocessing
    response = admin_client.post(f"/api/v1/content/{content_id}/reprocess")
    assert response.status_code == 200

def test_content_reprocess_unauthorized(authenticated_client: TestClient):
    """Test reprocessing content without admin privileges."""
    response = authenticated_client.post("/api/v1/content/some-id/reprocess")
    assert response.status_code == 403

# Chat Management Tests

def test_chat_query(authenticated_client: TestClient):
    """Test processing a chat query."""
    query_data = {
        "input": "What is the capital of France?",
        "child_id": "test-child-id",
        "language": "en"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=query_data)
    assert response.status_code == 200
    chat_response = response.json()
    assert "response" in chat_response
    assert "session_id" in chat_response
    assert "source_documents" in chat_response
    assert "confidence" in chat_response

def test_chat_query_with_session(authenticated_client: TestClient):
    """Test processing a chat query with an existing session."""
    # First create a session
    initial_query = {
        "input": "Hello",
        "child_id": "test-child-id",
        "language": "en"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=initial_query)
    session_id = response.json()["session_id"]
    
    # Use the session for a follow-up query
    follow_up_query = {
        "input": "What did I just say?",
        "session_id": session_id,
        "child_id": "test-child-id",
        "language": "en"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=follow_up_query)
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id

def test_chat_query_validation(authenticated_client: TestClient):
    """Test chat query input validation."""
    # Test with missing required fields
    response = authenticated_client.post("/api/v1/chat/query", json={})
    assert response.status_code == 422
    
    # Test with invalid language
    query_data = {
        "input": "Test",
        "child_id": "test-child-id",
        "language": "invalid"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=query_data)
    assert response.status_code == 400

def test_get_chat_sessions(authenticated_client: TestClient):
    """Test retrieving chat sessions."""
    # First create a session
    query_data = {
        "input": "Test session",
        "child_id": "test-child-id",
        "language": "en"
    }
    authenticated_client.post("/api/v1/chat/query", json=query_data)
    
    # Get sessions
    response = authenticated_client.get(
        "/api/v1/chat/sessions",
        params={"child_id": "test-child-id", "limit": 10}
    )
    assert response.status_code == 200
    sessions = response.json()
    assert isinstance(sessions, list)
    assert len(sessions) > 0

def test_get_session_messages(authenticated_client: TestClient):
    """Test retrieving messages for a chat session."""
    # First create a session with messages
    query_data = {
        "input": "Test message",
        "child_id": "test-child-id",
        "language": "en"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=query_data)
    session_id = response.json()["session_id"]
    
    # Get messages
    response = authenticated_client.get(
        f"/api/v1/chat/sessions/{session_id}",
        params={"limit": 50}
    )
    assert response.status_code == 200
    messages = response.json()
    assert isinstance(messages, list)
    assert len(messages) >= 2  # At least user message and assistant response

def test_get_session_messages_not_found(authenticated_client: TestClient):
    """Test retrieving messages for a non-existent session."""
    response = authenticated_client.get(
        "/api/v1/chat/sessions/nonexistent-id",
        params={"limit": 50}
    )
    assert response.status_code == 404

def test_provide_chat_feedback(authenticated_client: TestClient):
    """Test providing feedback for a chat message."""
    # First create a session with messages
    query_data = {
        "input": "Test feedback",
        "child_id": "test-child-id",
        "language": "en"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=query_data)
    session_id = response.json()["session_id"]
    
    # Get messages to find message_id
    response = authenticated_client.get(
        f"/api/v1/chat/sessions/{session_id}",
        params={"limit": 50}
    )
    messages = response.json()
    message_id = messages[0]["id"]  # Use the first message
    
    # Provide feedback
    feedback_data = {
        "message_id": message_id,
        "rating": 5,
        "comment": "Great response!"
    }
    response = authenticated_client.post("/api/v1/chat/feedback", json=feedback_data)
    assert response.status_code == 200
    feedback = response.json()
    assert feedback["message_id"] == message_id
    assert feedback["rating"] == 5

def test_delete_chat_session(authenticated_client: TestClient):
    """Test deleting a chat session."""
    # First create a session
    query_data = {
        "input": "Test delete",
        "child_id": "test-child-id",
        "language": "en"
    }
    response = authenticated_client.post("/api/v1/chat/query", json=query_data)
    session_id = response.json()["session_id"]
    
    # Delete the session
    response = authenticated_client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Session deleted successfully"
    
    # Verify session is deleted
    response = authenticated_client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == 404

def test_delete_chat_session_not_found(authenticated_client: TestClient):
    """Test deleting a non-existent chat session."""
    response = authenticated_client.delete("/api/v1/chat/sessions/nonexistent-id")
    assert response.status_code == 404
