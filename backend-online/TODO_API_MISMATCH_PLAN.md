# API Mismatch Implementation Plan

This document outlines the strategic plan to address missing or mismatched API endpoints and schema alignments between the `echo-nest-ai` frontend and the `backend-online` service.

## Reference Documents
- `API_ENDPOINTS.md`: Canonical list of all endpoints expected by the frontend.
- `mismatch-list.md`: Known mismatches and missing endpoints.

---

## 1. Groups Endpoints
**Status:** ✅ All implemented and registered

**Checklist:**
- [x] Implement `/groups` (GET/POST) for listing and creating groups
- [x] Implement `/groups/{id}` (GET) for group details
- [x] Implement `/groups/{id}/students` (GET) for students in a group
- [x] Implement `/groups/{id}/content` (GET) for content assigned to a group
- [x] Ensure all endpoints require authentication and proper access control
- [x] Align request/response schemas with frontend expectations

**Note:** All endpoints are production-grade, robust, and registered in the API router.

**Dependencies:** Group and student models, group membership logic

---

## 2. Students Endpoints
**Status:** ✅ All implemented and registered

**Checklist:**
- [x] Implement `/students` (GET/POST) for listing and creating students
- [x] Implement `/students/{id}` (PATCH/DELETE) for updating and deleting students
- [x] Implement `/students/{id}/groups` (GET) for groups a student belongs to
- [x] Implement `/students/group/{group_id}` (POST) to assign student to group
- [x] Implement `/students/group/{group_id}/{student_id}` (DELETE) to remove student from group
- [x] Ensure all endpoints require authentication and proper access control
- [x] Align request/response schemas with frontend expectations

**Note:** All endpoints are production-grade, robust, and registered in the API router.

**Dependencies:** Group endpoints, group membership logic

---

## 3. Dashboard Endpoints
**Status:** ✅ All implemented and registered

**Checklist:**
- [x] Implement `/dashboard/metrics/child/{id}/progress` (GET)
- [x] Implement `/dashboard/metrics/group/{id}/progress` (GET)
- [x] Implement `/dashboard/metrics/group/{id}/activities` (GET)
- [x] Implement `/dashboard/metrics/teacher/overview` (GET)
- [x] Ensure all endpoints require authentication and proper access control
- [x] Align request/response schemas with frontend expectations

**Note:** All endpoints are production-grade, robust, and registered in the API router.

**Dependencies:** Child, group, and content metrics logic

---

## 4. Notifications & Settings
**Status:** ✅ All implemented and registered

**Checklist:**
- [x] Implement `/events/poll` (GET) for notification polling
- [x] Implement `/settings/notifications` (PATCH) for updating notification settings
- [x] Ensure all endpoints require authentication
- [x] Align request/response schemas with frontend expectations

**Dependencies:** Notification/event model and logic

---

## 5. Feedback Stats & Schema Alignment
**Status:** ✅ All implemented and registered

**Checklist:**
- [x] Update `/feedback/stats` to support `tags` and `responseQuality` fields if required by frontend
- [x] Review and align all feedback and message schemas (fields, types, naming)
- [x] Add/adjust tests to ensure schema compatibility

**Dependencies:** None (schema and logic updates)

---

## 6. General
- [ ] Add/adjust tests for all new endpoints and schema changes
- [ ] Update OpenAPI/Swagger documentation
- [ ] Communicate changes to frontend team for integration

---

## Prioritization Strategy
1. **Groups & Students:** Core for organization/education mode; unblock related frontend features
2. **Dashboard:** High visibility for users; metrics and progress tracking
3. **Notifications:** User engagement and awareness
4. **Feedback/Schema:** Quality and analytics

---

## Special Considerations
- Ensure all endpoints are protected by authentication and proper authorization
- Use production-grade error handling and validation
- Only implement fields and logic actually used by the frontend
- Maintain backward compatibility where possible

---

*This plan should be updated as progress is made or as new requirements are discovered.* 