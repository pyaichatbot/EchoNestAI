import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from src.device_client import DeviceClient
from src.api_client import APIClient
from src.chat_manager import ChatManager
from src.language_manager import LanguageManager
from src.offline_server_manager import OfflineServerManager
from src.sync_manager import SyncManager

# Test Fixtures

@pytest.fixture
def mock_config():
    """Create a mock configuration file."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        config = {
            "api_url": "http://test-api.echonest.ai",
            "device_id": "test-device-001",
            "auth_token": "test-token",
            "offline_mode": False,
            "data_dir": str(Path(tempfile.gettempdir()) / "echonest_test_data"),
            "language": "en",
            "sync_interval": 300
        }
        json.dump(config, temp)
        temp.flush()
        return temp.name

@pytest.fixture
def device_client(mock_config):
    """Create a device client instance with mocked dependencies."""
    with patch('src.device_client.APIClient') as mock_api, \
         patch('src.device_client.ChatManager') as mock_chat, \
         patch('src.device_client.LanguageManager') as mock_lang, \
         patch('src.device_client.OfflineServerManager') as mock_server, \
         patch('src.device_client.SyncManager') as mock_sync:
        
        # Configure mock instances
        mock_api.return_value = Mock(spec=APIClient)
        mock_chat.return_value = Mock(spec=ChatManager)
        mock_lang.return_value = Mock(spec=LanguageManager)
        mock_server.return_value = Mock(spec=OfflineServerManager)
        mock_sync.return_value = Mock(spec=SyncManager)
        
        client = DeviceClient(config_path=mock_config)
        yield client
        
        # Cleanup
        if os.path.exists(mock_config):
            os.unlink(mock_config)

# Device Registration Tests

def test_device_registration(device_client):
    """Test device registration with valid code."""
    registration_code = "TEST-CODE-123"
    device_name = "Test Device"
    
    device_client.api_client.register_device.return_value = {
        "device_id": "test-device-001",
        "auth_token": "test-token",
        "status": "registered"
    }
    
    result = device_client.register_device(registration_code, device_name)
    assert result["status"] == "registered"
    assert result["device_id"] == "test-device-001"
    device_client.api_client.register_device.assert_called_once_with(
        registration_code, device_name
    )

def test_device_registration_invalid_code(device_client):
    """Test device registration with invalid code."""
    device_client.api_client.register_device.side_effect = Exception("Invalid registration code")
    
    with pytest.raises(Exception) as exc_info:
        device_client.register_device("INVALID-CODE", "Test Device")
    assert "Invalid registration code" in str(exc_info.value)

# Device Information Tests

def test_get_device_info(device_client):
    """Test retrieving device information."""
    expected_info = {
        "device_id": "test-device-001",
        "name": "Test Device",
        "status": "active",
        "last_sync": "2024-01-01T00:00:00Z"
    }
    
    device_client.api_client.get_device_info.return_value = expected_info
    
    info = device_client.get_device_info()
    assert info == expected_info
    device_client.api_client.get_device_info.assert_called_once()

# Chat Management Tests

def test_process_chat(device_client):
    """Test processing a chat query."""
    query = "What is the capital of France?"
    expected_response = {
        "response": "The capital of France is Paris.",
        "session_id": "test-session-001",
        "confidence": 0.95
    }
    
    device_client.chat_manager.process_query.return_value = expected_response
    
    result = device_client.process_chat(query=query)
    assert result == expected_response
    device_client.chat_manager.process_query.assert_called_once_with(
        query=query,
        session_id=None,
        language=None
    )

def test_process_chat_with_session(device_client):
    """Test processing a chat query with existing session."""
    query = "What did I just say?"
    session_id = "test-session-001"
    language = "en"
    
    expected_response = {
        "response": "You asked about the capital of France.",
        "session_id": session_id,
        "confidence": 0.95
    }
    
    device_client.chat_manager.process_query.return_value = expected_response
    
    result = device_client.process_chat(
        query=query,
        session_id=session_id,
        language=language
    )
    assert result == expected_response
    device_client.chat_manager.process_query.assert_called_once_with(
        query=query,
        session_id=session_id,
        language=language
    )

def test_get_chat_sessions(device_client):
    """Test retrieving chat sessions."""
    expected_sessions = [
        {"id": "session-1", "title": "Test Session 1"},
        {"id": "session-2", "title": "Test Session 2"}
    ]
    
    device_client.chat_manager.get_sessions.return_value = expected_sessions
    
    sessions = device_client.get_chat_sessions()
    assert sessions == expected_sessions
    device_client.chat_manager.get_sessions.assert_called_once()

def test_get_chat_messages(device_client):
    """Test retrieving chat messages for a session."""
    session_id = "test-session-001"
    limit = 50
    expected_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    device_client.chat_manager.get_messages.return_value = expected_messages
    
    messages = device_client.get_chat_messages(
        session_id=session_id,
        limit=limit
    )
    assert messages == expected_messages
    device_client.chat_manager.get_messages.assert_called_once_with(
        session_id=session_id,
        limit=limit
    )

# Voice Processing Tests

def test_process_voice(device_client):
    """Test processing a voice query."""
    audio_path = "test_audio.wav"
    session_id = "test-session-001"
    language = "en"
    
    expected_response = {
        "transcription": "This is a test voice message",
        "response": "I understood your message",
        "session_id": session_id
    }
    
    device_client.language_manager.process_voice.return_value = expected_response
    
    result = device_client.process_voice(
        audio_path=audio_path,
        session_id=session_id,
        language=language
    )
    assert result == expected_response
    device_client.language_manager.process_voice.assert_called_once_with(
        audio_path=audio_path,
        session_id=session_id,
        language=language
    )

# Synchronization Tests

def test_sync_now(device_client):
    """Test immediate synchronization."""
    device_client.sync_manager.sync_now.return_value = True
    
    result = device_client.sync_now()
    assert result is True
    device_client.sync_manager.sync_now.assert_called_once_with(force=False)

def test_sync_now_force(device_client):
    """Test forced synchronization."""
    device_client.sync_manager.sync_now.return_value = True
    
    result = device_client.sync_now(force=True)
    assert result is True
    device_client.sync_manager.sync_now.assert_called_once_with(force=True)

# Client Lifecycle Tests

def test_client_start(device_client):
    """Test client startup."""
    device_client.start()
    device_client.offline_server_manager.start.assert_called_once()
    device_client.sync_manager.start.assert_called_once()

def test_client_stop(device_client):
    """Test client shutdown."""
    device_client.stop()
    device_client.offline_server_manager.stop.assert_called_once()
    device_client.sync_manager.stop.assert_called_once()

# Error Handling Tests

def test_error_handling(device_client):
    """Test error handling in client operations."""
    device_client.chat_manager.process_query.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        device_client.process_chat(query="test")
    assert "Test error" in str(exc_info.value)

# Configuration Tests

def test_invalid_config_path():
    """Test client initialization with invalid config path."""
    with pytest.raises(FileNotFoundError):
        DeviceClient(config_path="nonexistent_config.json")

def test_missing_config_fields(mock_config):
    """Test client initialization with missing config fields."""
    with open(mock_config, 'r') as f:
        config = json.load(f)
    
    # Remove required field
    del config['api_url']
    
    with open(mock_config, 'w') as f:
        json.dump(config, f)
    
    with pytest.raises(ValueError) as exc_info:
        DeviceClient(config_path=mock_config)
    assert "Missing required configuration field" in str(exc_info.value) 