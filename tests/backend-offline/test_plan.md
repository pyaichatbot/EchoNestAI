# Backend Offline API Test Plan

This document outlines the tests for the EchoNest AI Offline Backend API.

## Objectives

- Verify the functionality of all exposed API endpoints.
- Ensure proper data validation and error handling.
- Test core offline features like local processing and data management.
- Check authentication and authorization mechanisms.

## Test Environment

- Python with pytest
- FastAPI TestClient
- Local SQLite database (or configured test database)
- Mocked external dependencies where necessary

## Test Suites

### 1. Health & Root Endpoints

- `test_root_endpoint`: Verify `/` returns correct status message.
- `test_health_endpoint`: Verify `/health` returns 200 OK and expected status/version.

### 2. Authentication (`/api/v1/auth`)

- `test_auth_register`: Test user registration with valid data.
- `test_auth_register_duplicate_email`: Test registration with existing email.
- `test_auth_register_validation`: Test input validation (email format, password complexity, missing fields).
- `test_auth_login`: Test successful login and token generation.
- `test_auth_login_invalid_credentials`: Test login with invalid credentials.
- `test_auth_login_inactive_user`: Test login with inactive user.
- `test_auth_get_current_user`: Test retrieving current user info with valid token.
- `test_auth_update_current_user`: Test updating current user info.
- `test_auth_forgot_password`: Test password reset request.
- `test_auth_reset_password`: Test password reset with valid token.
- `test_auth_reset_password_invalid_token`: Test password reset with invalid token.
- `test_auth_assign_role`: Test role assignment by admin.
- `test_auth_assign_role_unauthorized`: Test role assignment without admin privileges.

### 3. Protected Endpoints

- `test_protected_endpoints_require_auth`: Verify core endpoints (content, chat, devices, etc.) return 401 without a valid token.
- `test_invalid_token_access`: Test accessing protected routes with invalid/malformed tokens.

### 4. Content Management (`/api/v1/content`)

- `test_content_upload_local`: Test uploading content (text, potentially basic file types) for local storage.
- `test_content_list_local`: Test listing locally stored content.
- `test_content_get_local`: Test retrieving specific content by ID.
- `test_content_update_local`: Test updating metadata of local content.
- `test_content_delete_local`: Test deleting local content.
- `test_content_filtering_local`: Test filtering content by type, language (if applicable).
- `test_content_upload_validation`: Test validation for content upload (missing fields, invalid types).
- `test_content_not_found`: Test accessing non-existent content IDs (GET, PUT, DELETE).

### 5. Chat (`/api/v1/chat`)

- `test_chat_create_session_local`: Test creating a new local chat session.
- `test_chat_list_sessions_local`: Test listing local chat sessions.
- `test_chat_get_session_local`: Test retrieving a specific local chat session.
- `test_chat_send_message_local`: Test sending a message within a local session (potentially basic response generation).
- `test_chat_get_messages_local`: Test retrieving messages for a local session.
- `test_chat_delete_session_local`: Test deleting a local chat session.
- `test_chat_session_not_found`: Test operations on non-existent session IDs.
- `test_chat_message_validation`: Test validation for sending messages.

### 6. Device Management (`/api/v1/devices`)

- `test_device_register_local`: Test registering a new local device.
- `test_device_list_local`: Test listing registered local devices.
- `test_device_get_local`: Test retrieving a specific local device by ID.
- `test_device_update_local`: Test updating local device information.
- `test_device_delete_local`: Test deleting a local device.
- `test_device_validation`: Test input validation for registration and updates.
- `test_device_not_found`: Test accessing non-existent device IDs.

### 7. Language Features (`/api/v1/language`)

- `test_language_detect_local`: Test local language detection (if implemented).
- `test_language_translate_local`: Test local translation (if implemented).
- `test_language_tts_local`: Test local text-to-speech generation (if implemented).
- `test_language_supported_local`: Test listing locally supported languages/features.

### 8. Voice Features (`/api/v1/voice`)

- `test_voice_transcribe_local`: Test local voice transcription (upload and processing).
- `test_voice_get_transcription_result_local`: Test retrieving local transcription results.
- `test_voice_tts_local`: Test local voice text-to-speech (if different from language endpoint).
- `test_voice_chat_local`: Test local voice chat interaction.
- `test_voice_validation`: Test input validation for voice endpoints (file types, parameters).

## Notes

- Tests should clean up any created resources (users, content, sessions, devices).
- Use fixtures (`conftest.py`) for setting up the test client and managing dependencies.
- Focus on testing the offline capabilities and data handling specific to this backend. 