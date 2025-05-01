# API Mismatch List: echo-nest-ai vs backend-online (Comprehensive)

## 1. Endpoint Paths

| Feature/Area      | Frontend Endpoint(s) Used                | Backend Endpoint(s) Implemented           | Mismatch? | Notes |
|-------------------|------------------------------------------|-------------------------------------------|-----------|-------|
| **Auth**          | `/auth/login` (POST)                     | `/auth/login` (POST)                      | No        | Fully implemented |
|                   | `/auth/register` (POST)                  | `/auth/register` (POST)                   | No        | Fully implemented |
|                   | `/auth/token` (POST)                     | `/auth/token` (POST)                      | No        | Fully implemented |
|                   | `/auth/logout` (POST)                    | `/auth/logout` (POST)                     | No        | Fully implemented |
|                   | `/auth/verify-email` (POST/GET)          | `/auth/verify-email` (GET/POST)           | No        | Both methods supported |
| **Chat**          | `/chat/rag` (POST)                       | `/chat/rag` (POST)                        | No        | Fully implemented |
|                   | `/chat/rag/stream` (GET/POST)            | `/chat/rag/stream` (GET/POST)             | No        | Fully implemented |
|                   | `/chat/history` (GET)                    | `/chat/history` (GET)                     | No        | Fully implemented |
|                   | `/chat/session/{sessionId}` (GET)        | `/chat/session/{sessionId}` (GET)         | No        | Fully implemented |
|                   | `/chat/messages` (POST)                  | `/chat/messages` (POST)                   | No        | Fully implemented |
| **Chat Sessions** | `/chat/sessions` (GET/POST/DELETE)       | `/chat/sessions` (GET/POST/DELETE)        | No        | Fully implemented |
|                   | `/chat/sessions/{sessionId}` (GET/DELETE)| `/chat/sessions/{sessionId}` (GET/DELETE) | No        | Fully implemented |
|                   | `/chat/sessions/{sessionId}/messages`    | `/chat/sessions/{sessionId}/messages`     | No        | Fully implemented |
|                   | `/chat/sessions/child{childId}` (GET)    | `/chat/sessions/child{childId}` (GET)     | No        | Fully implemented |
| **Feedback**      | `/feedback/chat` (POST)                  | `/feedback/chat` (POST)                   | No        | Fully implemented |
|                   | `/feedback/stats` (GET)                  | `/feedback/stats` (GET)                   | Partial   | Tags/responseQuality not supported in backend |
|                   | `/feedback/flags` (GET)                  | `/feedback/flags` (GET)                   | No        | Fully implemented |
|                   | `/feedback/flags/{flagId}/resolve` (PATCH)| `/feedback/flags/{flagId}/resolve` (PATCH)| No        | Fully implemented |
|                   | `/feedback/flag` (POST)                  | `/feedback/flag` (POST)                   | No        | Fully implemented |
| **Content**       | `/content/list` (GET)                    | `/content` (GET)                          | **Partial**   | Path and filtering may differ; backend may need to add `/content/list` |
|                   | `/content/{id}` (GET)                    | `/content/{id}` (GET)                     | No        | Fully implemented |
|                   | `/content` (POST)                        | `/content` (POST)                         | No        | Fully implemented |
|                   | `/content/{id}` (PUT/DELETE)             | `/content/{id}` (PUT/DELETE)              | No        | Fully implemented |
|                   | `/content/upload` (POST)                 | `/content/upload` (POST)                  | No        | Fully implemented |
| **Children**      | `/children` (GET/POST)                   | _Not implemented_                         | **Yes**   | No children CRUD endpoints in backend |
|                   | `/children/{childId}` (GET/PATCH/DELETE) | _Not implemented_                         | **Yes**   | No children CRUD endpoints in backend |
| **Devices**       | `/devices` (GET/POST)                    | `/devices` (GET/POST)                     | No        | Fully implemented |
|                   | `/devices/{deviceId}` (GET/PUT/DELETE)   | `/devices/{deviceId}` (GET/PUT/DELETE)    | No        | Fully implemented |
| **Languages**     | `/languages/detect` (POST)               | `/languages/detect` (POST)                | No        | Fully implemented |
|                   | `/languages/supported` (GET)             | `/languages/supported` (GET)              | No        | Fully implemented |
|                   | `/languages/translate` (POST)            | `/languages/translate` (POST)             | No        | Fully implemented |
|                   | `/languages/tts` (POST)                  | `/languages/tts` (POST)                   | No        | Fully implemented |
|                   | `/languages/transcribe` (POST)           | `/languages/transcribe` (POST)            | No        | Fully implemented |
| **Voice**         | `/voice/tts` (POST)                      | `/voice/tts` (POST)                       | No        | Fully implemented |
|                   | `/voice/chat` (POST)                     | `/voice/chat` (POST)                      | No        | Fully implemented |
|                   | `/voice/transcribe` (POST)               | `/voice/transcribe` (POST)                | No        | Fully implemented |
| **Dashboard**     | `/dashboard` (GET)                       | `/dashboard` (GET)                        | No        | Fully implemented |
| **Events**        | `/events` (GET)                          | `/events` (GET)                           | No        | Fully implemented |

---

## 2. Request/Response Parameter and Schema Mismatches

- **Chat Feedback:**  Frontend sends extra fields (`tags`, `modelName`, `feedbackType`, `responseQuality`, `timestamp`) not supported by backend.
- **Content List:**  Frontend expects `/content/list` with advanced filtering; backend may only support `/content` or different filters.
- **Children:**  Frontend expects `/children` endpoints for CRUD; backend does not implement these.
- **Chat Message Persist:**  ~~Frontend calls `/chat/messages` (POST) to persist a message; backend does not implement this endpoint.~~ Now implemented.
- **Feedback Stats:**  Backend does not support tags or responseQuality fields; `commonTags` and `responseQualityBreakdown` are always empty.

---

## 3. Type/Schema Mismatches

- **Message/ChatMessage:**  Field names and types may differ (e.g., `senderId` vs `is_user`, `text` vs `content`, etc.).
- **ChatFeedback:**  Field names and extra fields differ as above.

---

## 4. Endpoints Missing or Not Fully Implemented in Backend

| Endpoint                        | Status      | Notes |
|----------------------------------|-------------|-------|
| `/chat/messages` (POST)          | **Implemented** | Now fully implemented with production-grade code and tests |
| `/content/list` (GET)            | **Partial** | Path and filtering may differ; backend may need to add `/content/list` |
| `/children` (GET/POST)           | **Missing** | No children CRUD endpoints in backend |
| `/children/{childId}` (GET/PATCH/DELETE) | **Missing** | No children CRUD endpoints in backend |

---

## 5. Recommendations

- **Implement missing endpoints**: ~~`/chat/messages`~~, `/content/list`, `/children` CRUD.
- **Align request/response schemas**: Standardize field names and types.
- **Document all endpoints and schemas** for both repos.
- **Review filtering and pagination** for `/content/list` and other list endpoints. 