# API Mismatch List: echo-nest-ai vs backend-online

## 1. Endpoint Paths

| Feature         | Frontend (echo-nest-ai) Endpoint         | Backend (backend-online) Endpoint         | Mismatch? | Notes |
|----------------|------------------------------------------|------------------------------------------|-----------|-------|
| Chat Query     | `/chat/rag` (POST)                       | `/chat/query` (POST)                     | **No**   | Frontend expects `/chat/rag`, backend exposes `/chat/query` |
| Chat Stream    | `/api/v1/chat/rag/stream` (POST/GET)      | `/chat/stream` (GET), `/chat/rag/stream` (GET/POST, now fixed) | **No**    | Backend now supports both GET and POST for `/api/v1/chat/rag/stream` and is fully tested. |
| Chat History   | `/chat/history` (GET) or `/chat/session/{sessionId}` (GET) | `/chat/history` (GET, now fixed)                        | **No**    | Backend now implements `/chat/history` with correct schema and pagination, fully tested. |
| Feedback       | `/feedback/chat` (POST)                   | `/feedback/chat` (POST)                  | No        | Paths match |
| Flag Content   | `/feedback/flag` (POST)                   | `/feedback/flag` (POST)                  | No        | Paths match |
| Feedback Stats | `/feedback/stats` (GET)                   | `/feedback/stats` (GET, now fixed)                        | **Partial** | Backend does not support tags or responseQuality fields; commonTags and responseQualityBreakdown are always empty. |
| Flag List      | `/feedback/flags` (GET)                   | `/feedback/flags` (GET, now fixed)                        | **No**    | Backend now implements `/feedback/flags` with correct schema and pagination, fully tested. |
| Flag Resolve   | `/feedback/flags/{flagId}/resolve` (PATCH)| `/feedback/flags/{flagId}/resolve` (PATCH, now fixed)         | **No**    | Backend now implements `/feedback/flags/{flagId}/resolve`, production-grade. |

## 2. Request/Response Parameter Mismatches

### Chat Query (POST)
- **Frontend:**
  - Expects `/chat/rag` (POST)
  - Body: `{ input, child_id, group_id, document_scope, language, session_id, context }`
- **Backend:**
  - Exposes `/chat/query` (POST)
  - Body: `{ input, child_id, group_id, document_scope, language, session_id }` (no `context`)
- **Mismatch:**
  - Endpoint path
  - `context` parameter is not supported by backend

### Chat History
- **Frontend:**
  - Calls `/chat/history` or `/chat/session/{sessionId}` (GET)
  - Expects `{ messages: Message[], hasMore: boolean, nextCursor?: string }`
- **Backend:**
  - No endpoint for chat history/messages
- **Mismatch:**
  - Endpoint missing

### Feedback (POST)
- **Frontend:**
  - Calls `/feedback/chat` (POST)
  - Body: `{ chatId, rating, comment, tags, modelName, feedbackType, responseQuality, timestamp }`
- **Backend:**
  - Expects `{ message_id, rating, comment, flagged, flag_reason }`
- **Mismatch:**
  - Field names differ (`chatId` vs `message_id`)
  - Extra fields in frontend (`tags`, `modelName`, `feedbackType`, `responseQuality`, `timestamp`)

### Flag Content (POST)
- **Frontend:**
  - Calls `/feedback/flag` (POST)
  - Body: `{ contentId, reason, flaggedBy, notes, timestamp }`
- **Backend:**
  - Expects `{ content_id, reason, flagged_by, notes }`
- **Mismatch:**
  - Field names differ (`contentId` vs `content_id`, `flaggedBy` vs `flagged_by`)
  - Extra field in frontend (`timestamp`)

### Feedback Stats, Flag List, Flag Resolve
- **Frontend:**
  - Calls `/feedback/stats`, `/feedback/flags`, `/feedback/flags/{flagId}/resolve`
- **Backend:**
  - No such endpoints
- **Mismatch:**
  - Endpoints missing

## 3. Type/Schema Mismatches
- **Message/ChatMessage**: Frontend expects fields like `id`, `senderId`, `text`, `timestamp`, `status`, `sourceDocuments`, `confidence`. Backend uses `id`, `session_id`, `is_user`, `content`, `created_at`, `source_documents`, `confidence`.
- **ChatFeedback**: Field names and extra fields differ as above.

---

## Recommendations
- Align endpoint paths and methods between frontend and backend.
- Add missing endpoints to backend (chat history, feedback stats, flag list, flag resolve).
- Standardize request/response field names and types.
- Document all endpoints and schemas for both repos.

## All endpoints in the current mismatch list are now addressed.

## Next endpoint to address: Flag Resolve 