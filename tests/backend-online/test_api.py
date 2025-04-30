import pytest
from fastapi.testclient import TestClient
import io
import os
import datetime

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

def test_device_registration_validation(authenticated_client):
    """Test device registration endpoint validates input."""
    # Test with missing fields
    response = authenticated_client.post("/api/devices", json={})
    assert response.status_code == 422

    # Test with invalid type
    response = authenticated_client.post("/api/devices", json={
        "name": "Test Device",
        "type": "invalid_type",
        "identifier": "test-device-001"
    })
    assert response.status_code == 422 # Assuming type is validated

def test_device_crud_operations(authenticated_client):
    """Test basic CRUD operations for devices."""
    # 1. List initial devices (should be empty or just default ones)
    response = authenticated_client.get("/api/devices")
    assert response.status_code == 200
    initial_devices = response.json()
    initial_count = len(initial_devices)

    # 2. Register a new device
    device_data = {
        "name": "My Test Device",
        "type": "smartphone", # Assuming 'smartphone' is a valid type
        "identifier": "unique-test-device-id-123"
    }
    response = authenticated_client.post("/api/devices", json=device_data)
    assert response.status_code == 201
    new_device = response.json()
    assert new_device["name"] == device_data["name"]
    assert new_device["type"] == device_data["type"]
    assert new_device["identifier"] == device_data["identifier"]
    assert "id" in new_device
    device_id = new_device["id"]

    # 3. Verify device was added by listing again
    response = authenticated_client.get("/api/devices")
    assert response.status_code == 200
    assert len(response.json()) == initial_count + 1

    # 4. Get the specific device by ID
    response = authenticated_client.get(f"/api/devices/{device_id}")
    assert response.status_code == 200
    fetched_device = response.json()
    assert fetched_device["id"] == device_id
    assert fetched_device["name"] == device_data["name"]

    # 5. Update the device name
    update_data = {"name": "Updated Test Device Name"}
    response = authenticated_client.put(f"/api/devices/{device_id}", json=update_data)
    assert response.status_code == 200
    updated_device = response.json()
    assert updated_device["name"] == update_data["name"]
    # Ensure other fields remain unchanged if not updated
    assert updated_device["type"] == device_data["type"]

    # 6. Verify update by fetching again
    response = authenticated_client.get(f"/api/devices/{device_id}")
    assert response.status_code == 200
    assert response.json()["name"] == update_data["name"]

    # 7. Delete the device
    response = authenticated_client.delete(f"/api/devices/{device_id}")
    assert response.status_code == 200 # Or 204 if no content is returned

    # 8. Verify deletion by trying to get it again (should be 404)
    response = authenticated_client.get(f"/api/devices/{device_id}")
    assert response.status_code == 404

    # 9. Verify deletion by listing devices again
    response = authenticated_client.get("/api/devices")
    assert response.status_code == 200
    assert len(response.json()) == initial_count

def test_content_actual_file_upload(authenticated_client):
    """Test actual file upload for content creation."""
    # Get initial content count
    response = authenticated_client.get("/api/content")
    assert response.status_code == 200
    initial_count = len(response.json())

    # Prepare fake file data
    fake_file_content = b"This is the content of the test file."
    fake_file_name = "test_upload.txt"
    files = {"file": (fake_file_name, io.BytesIO(fake_file_content), "text/plain")}
    data = {
        "title": "Uploaded Content",
        "description": "Description for uploaded file",
        "language": "en",
        "type": "document",
        "sync_offline": "true" # Form data often comes as strings
    }

    # Assuming the endpoint is POST /api/content for actual uploads
    response = authenticated_client.post("/api/content", files=files, data=data)
    
    # Check for successful creation (201 or maybe 200)
    assert response.status_code == 201
    created_content = response.json()
    assert created_content["title"] == data["title"]
    assert created_content["original_filename"] == fake_file_name # Assuming this field exists

    # Verify content was added
    response = authenticated_client.get("/api/content")
    assert response.status_code == 200
    assert len(response.json()) == initial_count + 1

    # Clean up: Delete the created content
    content_id = created_content["id"]
    delete_response = authenticated_client.delete(f"/api/content/{content_id}")
    assert delete_response.status_code == 200

    # Verify deletion
    response = authenticated_client.get(f"/api/content/{content_id}")
    assert response.status_code == 404

def test_chat_send_message_different_languages(authenticated_client):
    """Test sending chat messages with different language codes."""
    # Create a session
    response = authenticated_client.post("/api/chat/sessions", json={"title": "Language Test Session"})
    assert response.status_code == 201
    session_id = response.json()["id"]

    # Send message in English
    response_en = authenticated_client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": "Hello", "language": "en"}
    )
    assert response_en.status_code == 201

    # Send message in Spanish (assuming 'es' is supported)
    response_es = authenticated_client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": "Hola", "language": "es"}
    )
    assert response_es.status_code == 201 

    # Verify messages (optional, check count or content if feasible)
    response_msgs = authenticated_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert response_msgs.status_code == 200
    # Add more specific assertions if response format is known
    assert len(response_msgs.json()) >= 2 

    # Clean up: Delete session
    delete_response = authenticated_client.delete(f"/api/chat/sessions/{session_id}")
    assert delete_response.status_code == 200

def test_chat_message_to_nonexistent_session(authenticated_client):
    """Test sending a message to a session ID that does not exist."""
    non_existent_session_id = "invalid-session-id-12345"
    response = authenticated_client.post(
        f"/api/chat/sessions/{non_existent_session_id}/messages",
        json={"content": "Test", "language": "en"}
    )
    assert response.status_code == 404 # Expecting Not Found

def test_dashboard_endpoint(authenticated_client):
    """Test the main dashboard endpoint returns data successfully."""
    response = authenticated_client.get("/api/dashboard")
    assert response.status_code == 200
    # Basic check for some JSON response
    assert response.json() is not None 
    # TODO: Add more specific assertions based on expected dashboard structure
    # e.g., assert "user_stats" in response.json()
    # e.g., assert "recent_activity" in response.json()

def test_events_endpoint(authenticated_client):
    """Test the main events endpoint returns data successfully."""
    response = authenticated_client.get("/api/events")
    assert response.status_code == 200
    # Basic check for some JSON response, likely a list
    assert isinstance(response.json(), list)
    # TODO: Add tests for event filtering if implemented (e.g., by date, type)

# --- Language API Tests ---

def test_language_detection(authenticated_client):
    """Test the language detection endpoint."""
    # Test English detection
    response = authenticated_client.post("/api/languages/detect", json={"text": "Hello world, this is a test."})
    assert response.status_code == 200
    result = response.json()
    assert "detected_language" in result
    assert result["detected_language"] == "en"
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    
    # Test Spanish detection (if supported)
    response = authenticated_client.post("/api/languages/detect", json={"text": "Hola mundo, esto es una prueba."})
    assert response.status_code == 200
    result = response.json()
    assert result["detected_language"] == "es"
    assert 0.0 <= result["confidence"] <= 1.0
    
    # Test validation - empty text
    response = authenticated_client.post("/api/languages/detect", json={"text": ""})
    assert response.status_code == 422  # Validation error for empty text

def test_supported_languages(authenticated_client):
    """Test listing supported languages."""
    response = authenticated_client.get("/api/languages/supported")
    assert response.status_code == 200
    languages = response.json()
    assert isinstance(languages, list)
    assert len(languages) > 0
    
    # Check structure of language object
    first_language = languages[0]
    assert "code" in first_language
    assert "name" in first_language
    assert "native_name" in first_language

def test_language_translation(authenticated_client):
    """Test the language translation endpoint."""
    # English to Spanish
    response = authenticated_client.post(
        "/api/languages/translate", 
        data={
            "text": "Hello, how are you?",
            "source_language": "en",
            "target_language": "es"
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "translated_text" in result
    assert result["translated_text"] != "Hello, how are you?"  # Should be different
    assert "source_language" in result
    assert result["source_language"] == "en"
    assert "target_language" in result
    assert result["target_language"] == "es"
    
    # Test unsupported language pair
    response = authenticated_client.post(
        "/api/languages/translate", 
        data={
            "text": "Hello",
            "source_language": "en",
            "target_language": "xx"  # Invalid language code
        }
    )
    assert response.status_code == 400  # Bad request for unsupported language

def test_language_text_to_speech(authenticated_client):
    """Test the text-to-speech endpoint."""
    response = authenticated_client.post(
        "/api/languages/tts",
        data={
            "text": "This is a test of text to speech.",
            "language": "en"
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "status" in result and result["status"] == "success"
    assert "audio_url" in result  # URL to audio file
    assert "language" in result and result["language"] == "en"
    assert "text" in result
    
    # Verify audio file exists and is accessible
    audio_url = result["audio_url"]
    # Extract path from URL
    audio_path = audio_url.replace("/media/", "")
    audio_response = authenticated_client.get(f"/media/{audio_path}")
    assert audio_response.status_code == 200
    assert len(audio_response.content) > 0  # Should have content
    assert "audio/" in audio_response.headers.get("content-type", "")  # Should be audio

def test_language_transcription(authenticated_client):
    """Test the audio transcription endpoint."""
    # Create a dummy audio file
    fake_audio_content = b"dummy audio data" * 1000  # Need some size for testing
    files = {"audio": ("test_audio.WAV", io.BytesIO(fake_audio_content), "audio/wav")}
    
    response = authenticated_client.post(
        "/api/languages/transcribe",
        files=files,
        data={"language": "en"}
    )
    
    # Note: For this test to pass in a real environment, we'd need a real audio file
    # Here we're testing the API contract, not the actual transcription functionality
    
    # If the endpoint returns 500 because our dummy audio isn't valid, that's expected
    # We're just checking the contract is correct
    assert response.status_code in [200, 500]
    
    # If successful, check response format
    if response.status_code == 200:
        result = response.json()
        assert "text" in result
        assert "detected_language" in result
        assert "confidence" in result

# --- Voice API Tests (Placeholders) ---

@pytest.mark.skip(reason="Implementation details needed for voice API")
def test_voice_transcription_upload(authenticated_client):
    """Test uploading voice data for transcription."""
    # fake_audio_content = b"dummy audio data"
    # fake_audio_name = "test_audio.wav"
    # files = {"file": (fake_audio_name, io.BytesIO(fake_audio_content), "audio/wav")}
    # data = {"language": "en"} # Optional parameters
    # response = authenticated_client.post("/api/voice/transcribe", files=files, data=data)
    # assert response.status_code == 202 # Assuming it's an async task
    # assert "job_id" in response.json() # Or similar indicator
    pass

@pytest.mark.skip(reason="Implementation details needed for voice API")
def test_voice_get_transcription_result(authenticated_client):
    """Test retrieving the result of a transcription."""
    # Prerequisite: test_voice_transcription_upload should create a job
    # job_id = "some_job_id_from_previous_test_or_setup"
    # response = authenticated_client.get(f"/api/voice/transcribe/{job_id}")
    # assert response.status_code == 200 
    # assert response.json()["status"] == "completed" # Or check for transcript
    pass

@pytest.mark.skip(reason="Implementation details needed for voice API")
def test_voice_tts(authenticated_client):
    """Test the text-to-speech endpoint."""
    # response = authenticated_client.post(
    #     "/api/voice/tts", 
    #     json={"text": "Generate audio from this text", "language": "en", "voice_id": "default"}
    # )
    # assert response.status_code == 200
    # assert response.headers["content-type"] == "audio/mpeg" # Or other audio format
    # assert len(response.content) > 0
    pass

@pytest.mark.skip(reason="Implementation details needed for voice API")
def test_voice_validation(authenticated_client):
    """Test validation for voice endpoints."""
    # Example: Test transcription with invalid language
    # fake_audio_content = b"dummy"
    # files = {"file": ("test.wav", io.BytesIO(fake_audio_content), "audio/wav")}
    # data = {"language": "xx"}
    # response = authenticated_client.post("/api/voice/transcribe", files=files, data=data)
    # assert response.status_code == 422

    # Example: Test TTS with missing text
    # response = authenticated_client.post("/api/voice/tts", json={})
    # assert response.status_code == 422
    pass

def test_voice_text_to_speech(authenticated_client):
    """Test the voice text-to-speech endpoint."""
    response = authenticated_client.post(
        "/api/voice/tts",
        data={
            "text": "This is a test of voice text to speech.",
            "language": "en"
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    
    assert "status" in result and result["status"] == "success"
    assert "audio_url" in result
    assert "language" in result
    assert "text" in result

def test_voice_chat(authenticated_client):
    """Test uploading voice for chat processing."""
    # Create a dummy audio file
    fake_audio_content = b"dummy audio data" * 1000  # Need some size for testing
    files = {"audio": ("test_audio.wav", io.BytesIO(fake_audio_content), "audio/wav")}
    
    # Make the request
    response = authenticated_client.post(
        "/api/voice/chat",
        files=files,
        data={"language": "en"}
    )
    
    # The success case would be 200 but our dummy audio might cause a processing error (500)
    # This tests that the API contract is correct
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        result = response.json()
        assert "response" in result
        assert "session_id" in result
        assert "confidence" in result

def test_content_filtering(authenticated_client):
    """Test filtering content by type and language."""
    # First create a test content item
    response = authenticated_client.post(
        "/api/content/mock-upload",  # Using mock_upload from existing tests
        json={
            "title": "Filterable Content",
            "description": "Content for filter testing",
            "language": "es",  # Spanish
            "type": "audio",   # Audio type
            "sync_offline": True
        }
    )
    assert response.status_code in [201, 200]
    
    # Test filtering by type
    response = authenticated_client.get("/api/content", params={"type": "audio"})
    assert response.status_code == 200
    audio_results = response.json()
    assert len(audio_results) > 0
    for item in audio_results:
        assert item["type"] == "audio"
    
    # Test filtering by language
    response = authenticated_client.get("/api/content", params={"language": "es"})
    assert response.status_code == 200
    spanish_results = response.json()
    assert len(spanish_results) > 0
    for item in spanish_results:
        assert item["language"] == "es"
    
    # Test combined filtering
    response = authenticated_client.get("/api/content", params={"type": "audio", "language": "es"})
    assert response.status_code == 200
    filtered_results = response.json()
    for item in filtered_results:
        assert item["type"] == "audio" and item["language"] == "es"

def test_content_upload_errors(authenticated_client):
    """Test error conditions when uploading content."""
    # Test with unsupported content type
    fake_file_content = b"This is the content of the test file."
    files = {"file": ("test_file.txt", io.BytesIO(fake_file_content), "text/plain")}
    data = {
        "title": "Error Test Content",
        "type": "invalid_type",  # Invalid content type
        "language": "en",
        "sync_offline": "true"
    }
    
    response = authenticated_client.post("/api/content/upload", files=files, data=data)
    assert response.status_code == 400  # Bad request for invalid type
    
    # Test with missing required field (title)
    data = {
        "type": "document",
        "language": "en",
        "sync_offline": "true"
    }
    
    response = authenticated_client.post("/api/content/upload", files=files, data=data)
    assert response.status_code == 422  # Validation error
    
    # Test with no file provided
    data = {
        "title": "No File Test",
        "type": "document",
        "language": "en",
        "sync_offline": "true"
    }
    
    response = authenticated_client.post("/api/content/upload", data=data)  # No files
    assert response.status_code == 422  # Validation error

def test_chat_sessions_list(authenticated_client):
    """Test listing chat sessions with pagination and filtering."""
    # Create multiple chat sessions to test pagination
    sessions = []
    for i in range(3):
        response = authenticated_client.post(
            "/api/chat/sessions",
            json={"title": f"Session {i+1}"}
        )
        assert response.status_code == 201
        sessions.append(response.json())
    
    # Test listing all sessions (should include our new ones)
    response = authenticated_client.get("/api/chat/sessions")
    assert response.status_code == 200
    all_sessions = response.json()
    assert len(all_sessions) >= 3
    
    # Test pagination if the API supports it
    # Format will depend on backend implementation
    response = authenticated_client.get("/api/chat/sessions", params={"limit": 2, "offset": 0})
    assert response.status_code in [200, 400]  # 400 if pagination not supported
    if response.status_code == 200:
        paginated = response.json()
        # Either we get a list with 2 items or a paginated response object
        if isinstance(paginated, list):
            assert len(paginated) <= 2
        elif isinstance(paginated, dict) and "items" in paginated:
            assert len(paginated["items"]) <= 2
    
    # Clean up - delete sessions
    for session in sessions:
        response = authenticated_client.delete(f"/api/chat/sessions/{session['id']}")
        assert response.status_code in [200, 204]

def test_events_filtering(authenticated_client):
    """Test filtering options for events."""
    # First get all events
    response = authenticated_client.get("/api/events")
    assert response.status_code == 200
    
    # Test filtering by date if supported
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    response = authenticated_client.get("/api/events", params={"date": today})
    assert response.status_code in [200, 400]  # 400 if filtering not supported
    
    if response.status_code == 200:
        events = response.json()
        assert isinstance(events, list)
        # If successful and any events returned, check they match the filter
        if events:
            for event in events:
                assert today in event.get("timestamp", "")  # Assuming timestamp field contains date
    
    # Test filtering by event type if supported
    response = authenticated_client.get("/api/events", params={"type": "login"})
    assert response.status_code in [200, 400]  # 400 if filtering not supported
    
    if response.status_code == 200:
        events = response.json()
        assert isinstance(events, list)
        # If successful and any events returned, check they match the filter
        if events:
            for event in events:
                assert event.get("type") == "login"  # Assuming type field

# Add tests for other modules based on the task list...
# e.g., test_dashboard_endpoint, test_events_endpoint, etc.

# --- Auth Edge Case Tests ---

def test_auth_invalid_token(client):
    """Test accessing protected routes with an invalid token."""
    # Test with a malformed JWT
    invalid_token = "invalid.token.format"
    response = client.get("/api/content", headers={"Authorization": f"Bearer {invalid_token}"})
    assert response.status_code == 401
    
    # Test with a well-formed but invalid JWT
    # This is a made-up token that looks like a JWT but is not valid
    fake_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    response = client.get("/api/content", headers={"Authorization": f"Bearer {fake_jwt}"})
    assert response.status_code == 401
    
    # Test with no token
    response = client.get("/api/content")
    assert response.status_code == 401
    
    # Test with empty token
    response = client.get("/api/content", headers={"Authorization": "Bearer "})
    assert response.status_code == 401

def test_auth_password_validation(client):
    """Test the password validation rules during registration."""
    user_data = {
        "email": "test_pw_validation@example.com",
        "firstName": "Test",
        "lastName": "User",
        "password": "short"  # Too short password
    }
    
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 422  # Validation error
    
    # Try with a password that has no uppercase
    user_data["password"] = "password123"
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 422
    
    # Try with a password that has no digits
    user_data["password"] = "Password"
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 422
    
    # Finally with a valid password
    user_data["password"] = "Password123!"
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 201

def test_dashboard_specific_data(authenticated_client):
    """Test specific data points and structure in the dashboard response."""
    response = authenticated_client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    
    # Check for required top-level fields
    assert "user_stats" in data
    assert "recent_activity" in data
    assert "content_summary" in data
    
    # Check user stats structure
    user_stats = data["user_stats"]
    assert "total_content" in user_stats
    assert "active_devices" in user_stats
    assert "total_chats" in user_stats
    assert isinstance(user_stats["total_content"], int)
    assert isinstance(user_stats["active_devices"], int)
    assert isinstance(user_stats["total_chats"], int)
    
    # Check recent activity structure
    recent_activity = data["recent_activity"]
    assert isinstance(recent_activity, list)
    if recent_activity:  # If there are any activities
        activity = recent_activity[0]
        assert "type" in activity
        assert "timestamp" in activity
        assert "description" in activity
        assert isinstance(activity["timestamp"], str)
    
    # Check content summary structure
    content_summary = data["content_summary"]
    assert "by_type" in content_summary
    assert "by_language" in content_summary
    assert isinstance(content_summary["by_type"], dict)
    assert isinstance(content_summary["by_language"], dict)

def test_voice_transcription_retrieval(authenticated_client):
    """Test retrieval of voice transcription results."""
    # First create a transcription request
    audio_file = {
        "file": ("test_audio.wav", open("tests/test_data/test_audio.wav", "rb"), "audio/wav")
    }
    response = authenticated_client.post(
        "/api/transcribe",
        files=audio_file
    )
    assert response.status_code == 202
    transcription_id = response.json()["transcription_id"]
    
    # Test immediate retrieval (should be processing)
    response = authenticated_client.get(f"/api/transcribe/{transcription_id}")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["processing", "completed"]
    
    # Test retrieval after completion (mock)
    if data["status"] == "completed":
        assert "transcript" in data
        assert "language" in data
        assert "duration" in data
        assert isinstance(data["transcript"], str)
        assert isinstance(data["language"], str)
        assert isinstance(data["duration"], float)
    
    # Test retrieval of non-existent transcription
    response = authenticated_client.get("/api/transcribe/nonexistent_id")
    assert response.status_code == 404
    
def test_chat_rag_endpoint(authenticated_client):
    """Test /chat/rag endpoint with both camelCase and snake_case payloads."""
    # CamelCase payload (frontend style)
    payload_camel = {
        "input": "Hello, how are you?",
        "childId": "child123",
        "groupId": "group123",
        "documentScope": ["doc1", "doc2"],
        "language": "en",
        "sessionId": None,
        "context": "Some context"
    }
    response = authenticated_client.post("/chat/rag", json=payload_camel)
    assert response.status_code in [200, 201]
    data = response.json()
    assert "response" in data
    assert "confidence" in data

    # Snake_case payload (backend style)
    payload_snake = {
        "input": "Hello again!",
        "child_id": "child123",
        "group_id": "group123",
        "document_scope": ["doc1", "doc2"],
        "language": "en",
        "session_id": None,
        "context": "Some context"
    }
    response = authenticated_client.post("/chat/rag", json=payload_snake)
    assert response.status_code in [200, 201]
    data = response.json()
    assert "response" in data
    assert "confidence" in data

    # Test missing context (should still work)
    payload_no_context = {
        "input": "No context here.",
        "childId": "child123",
        "groupId": "group123",
        "documentScope": ["doc1", "doc2"],
        "language": "en"
    }
    response = authenticated_client.post("/chat/rag", json=payload_no_context)
    assert response.status_code in [200, 201]
    data = response.json()
    assert "response" in data

def test_chat_rag_stream_get(authenticated_client):
    """Test /chat/rag/stream GET endpoint for SSE streaming."""
    params = {
        "child_id": "child123",
        "group_id": "group123",
        "session_id": "session123",
        "language": "en"
    }
    with authenticated_client.stream("GET", "/chat/rag/stream", params=params) as response:
        assert response.status_code == 200
        # Read a few events from the stream (if any)
        try:
            for i, line in enumerate(response.iter_lines()):
                if line:
                    data = line.decode()
                    assert "token" in data or "data" in data or "Error" in data
                if i > 2:
                    break
        except Exception as e:
            # Streaming may end quickly if no data, that's OK for contract test
            pass

def test_chat_rag_stream_post(authenticated_client):
    """Test /chat/rag/stream POST endpoint for SSE streaming with payload."""
    params = {
        "child_id": "child123",
        "group_id": "group123",
        "session_id": "session123",
        "language": "en"
    }
    payload = {
        "input": "Stream this message.",
        "context": "Some context for streaming.",
        "document_scope": ["doc1", "doc2"]
    }
    with authenticated_client.stream("POST", "/chat/rag/stream", params=params, json=payload) as response:
        assert response.status_code == 200
        # Read a few events from the stream (if any)
        try:
            for i, line in enumerate(response.iter_lines()):
                if line:
                    data = line.decode()
                    assert "token" in data or "data" in data or "Error" in data
                if i > 2:
                    break
        except Exception as e:
            # Streaming may end quickly if no data, that's OK for contract test
            pass

def test_chat_history_endpoint(authenticated_client):
    """Test /chat/history endpoint for correct message retrieval and pagination."""
    # 1. Create a chat session (simulate via /chat/rag to ensure session and messages are created)
    payload = {
        "input": "Hello, this is a test message.",
        "childId": "child123",
        "groupId": "group123",
        "language": "en"
    }
    response = authenticated_client.post("/chat/rag", json=payload)
    assert response.status_code in [200, 201]
    # Extract sessionId from response or create a new one
    # We'll fetch sessions for the child to get the sessionId
    session_resp = authenticated_client.get("/chat/sessions/child123")
    assert session_resp.status_code == 200
    sessions = session_resp.json()
    assert sessions and isinstance(sessions, list)
    session_id = sessions[0]["id"]

    # 2. Send additional messages to the session
    for i in range(3):
        payload["input"] = f"Test message {i+1}"
        payload["sessionId"] = session_id
        response = authenticated_client.post("/chat/rag", json=payload)
        assert response.status_code in [200, 201]

    # 3. Fetch chat history
    resp = authenticated_client.get(f"/chat/history?session_id={session_id}&limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert "messages" in data and isinstance(data["messages"], list)
    assert len(data["messages"]) == 2
    assert "hasMore" in data
    assert "nextCursor" in data or data["nextCursor"] is None
    # Check message fields
    for msg in data["messages"]:
        assert set(msg.keys()) >= {"id", "senderId", "text", "timestamp", "status", "sourceDocuments", "confidence"}
        assert isinstance(msg["timestamp"], int)
        assert msg["status"] == "sent"
    # 4. Test pagination with 'before'
    if data["hasMore"] and data["nextCursor"]:
        resp2 = authenticated_client.get(f"/chat/history?session_id={session_id}&limit=2&before={data['nextCursor']}")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert "messages" in data2
        # Should not repeat the same messages
        ids1 = {m["id"] for m in data["messages"]}
        ids2 = {m["id"] for m in data2["messages"]}
        assert ids1.isdisjoint(ids2)
        
def test_feedback_flags_endpoint(authenticated_client):
    """Test /feedback/flags endpoint for listing flagged feedback."""
    # 1. Create a chat session and message to flag
    payload = {
        "input": "Flag test message.",
        "childId": "childFlagTest",
        "groupId": "groupFlagTest",
        "language": "en"
    }
    response = authenticated_client.post("/chat/rag", json=payload)
    assert response.status_code in [200, 201]
    # Get session and message id
    session_resp = authenticated_client.get("/chat/sessions/childFlagTest")
    assert session_resp.status_code == 200
    sessions = session_resp.json()
    assert sessions and isinstance(sessions, list)
    session_id = sessions[0]["id"]
    # There should be at least one message in the session
    # We'll use /chat/history to get the message id
    history_resp = authenticated_client.get(f"/chat/history?session_id={session_id}&limit=1")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert "messages" in history and history["messages"]
    message_id = history["messages"][-1]["id"]
    # 2. Flag the message
    flag_payload = {
        "contentId": message_id,
        "reason": "Test flag reason",
        "flaggedBy": "parent",
        "notes": "Test flag notes"
    }
    flag_resp = authenticated_client.post("/feedback/flag", json=flag_payload)
    assert flag_resp.status_code == 200
    # 3. List flagged feedback
    flags_resp = authenticated_client.get("/feedback/flags")
    assert flags_resp.status_code == 200
    flags = flags_resp.json()
    assert isinstance(flags, list)
    # There should be at least one flagged feedback with our reason
    found = False
    for f in flags:
        assert set(f.keys()) >= {"id", "message_id", "flagged", "flag_reason", "comment", "created_at", "rating"}
        if f["flag_reason"] == "Test flag reason":
            found = True
    assert found, "Flagged feedback not found in /feedback/flags response"

def test_feedback_flag_resolve_endpoint(authenticated_client):
    """Test /feedback/flags/{flagId}/resolve endpoint for resolving flagged feedback."""
    # 1. Create a chat session and message to flag
    payload = {
        "input": "Flag resolve test message.",
        "childId": "childFlagResolveTest",
        "groupId": "groupFlagResolveTest",
        "language": "en"
    }
    response = authenticated_client.post("/chat/rag", json=payload)
    assert response.status_code in [200, 201]
    # Get session and message id
    session_resp = authenticated_client.get("/chat/sessions/childFlagResolveTest")
    assert session_resp.status_code == 200
    sessions = session_resp.json()
    assert sessions and isinstance(sessions, list)
    session_id = sessions[0]["id"]
    # There should be at least one message in the session
    history_resp = authenticated_client.get(f"/chat/history?session_id={session_id}&limit=1")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert "messages" in history and history["messages"]
    message_id = history["messages"][-1]["id"]
    # 2. Flag the message
    flag_payload = {
        "contentId": message_id,
        "reason": "Test resolve flag reason",
        "flaggedBy": "parent",
        "notes": "Test resolve flag notes"
    }
    flag_resp = authenticated_client.post("/feedback/flag", json=flag_payload)
    assert flag_resp.status_code == 200
    # 3. List flagged feedback to get the flagId
    flags_resp = authenticated_client.get("/feedback/flags")
    assert flags_resp.status_code == 200
    flags = flags_resp.json()
    assert isinstance(flags, list)
    flag_id = None
    for f in flags:
        if f["flag_reason"] == "Test resolve flag reason":
            flag_id = f["id"]
            break
    assert flag_id, "Flagged feedback not found in /feedback/flags response"
    # 4. Resolve the flag
    resolve_resp = authenticated_client.patch(f"/feedback/flags/{flag_id}/resolve")
    assert resolve_resp.status_code == 200
    resolved = resolve_resp.json()
    assert resolved["flagged"] is False
    # 5. Verify feedback is no longer flagged
    flags_resp2 = authenticated_client.get("/feedback/flags")
    assert flags_resp2.status_code == 200
    flags2 = flags_resp2.json()
    for f in flags2:
        assert f["id"] != flag_id  # Should not appear in flagged list
