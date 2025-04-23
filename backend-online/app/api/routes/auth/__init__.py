from fastapi import APIRouter
from app.api.routes.auth.login import router as login_router
from app.api.routes.auth.registration import router as registration_router
from app.api.routes.auth.password import router as password_router
from app.api.routes.auth.verification import router as verification_router
from app.api.routes.auth.admin import router as admin_router

router = APIRouter(prefix="/auth", tags=["authentication"])

# Include all auth-related routers
router.include_router(login_router)
router.include_router(registration_router)
router.include_router(password_router)
router.include_router(verification_router)
router.include_router(admin_router)
