# API Test Plan

This document outlines the remaining tests to be implemented for the backend API.

## Task List

-   **Content:**
    -   [x] Test actual file upload for `/api/content` using `multipart/form-data`.
    -   [x] Test content filtering/search if available (e.g., by language, type).
    -   [x] Test specific error conditions (e.g., uploading invalid file type, exceeding size limits if applicable).
-   **Chat:**
    -   [x] Test fetching chat sessions with pagination or filtering if implemented.
    -   [x] Test sending messages with different languages.
    -   [x] Test potential error states (e.g., sending message to non-existent session).
-   **Dashboard:**
    -   [x] Test the main dashboard endpoint (assuming `/api/dashboard`) to ensure it returns data with status code 200 for authenticated users.
    -   [ ] Test any specific data points or structures expected in the dashboard response.
-   **Devices:**
    -   [x] Test listing devices (`GET /api/devices`).
    -   [x] Test registering a new device (`POST /api/devices`).
    -   [x] Test getting a specific device (`GET /api/devices/{device_id}`).
    -   [x] Test updating a device (`PUT /api/devices/{device_id}`).
    -   [x] Test deleting a device (`DELETE /api/devices/{device_id}`).
    -   [x] Test validation for device registration/update.
-   **Events:**
    -   [x] Test the events endpoint (`GET /api/events` or similar) to ensure it returns data with status code 200 for authenticated users.
    -   [x] Test any filtering options for events (e.g., by date, type).
-   **Language:**
    -   [x] Test language detection endpoint (assuming one exists, e.g., `POST /api/language/detect`).
    -   [x] Test translation endpoint (assuming one exists, e.g., `POST /api/language/translate`).
    -   [x] Test validation for language endpoints (e.g., invalid language codes, missing text).
-   **Voice:**
    -   [x] Test voice upload/processing endpoint (e.g., `POST /api/voice/transcribe`). Requires simulating file upload.
    -   [ ] Test retrieving transcription results.
    -   [x] Test text-to-speech endpoint (e.g., `POST /api/voice/tts`).
    -   [x] Test validation for voice endpoints.
-   **Auth Edge Cases:**
    -   [ ] Test token refresh mechanism (if implemented).
    -   [x] Test accessing protected routes with expired/invalid tokens (should result in 401). 