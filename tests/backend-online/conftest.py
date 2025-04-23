import pytest
import os
import sys
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Add the application directory to the Python path
sys.path.append("/app")

from app.main import app
from app.db.database import get_db
from app.db.models.models import Base
from app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@postgres-test:5432/test_db"

# Create async engine for tests
engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
    poolclass=NullPool,
)

# Create async session for tests
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Override the get_db dependency
async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

# Fixture to create and drop test database tables
@pytest.fixture(scope="module")
async def setup_database():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# Fixture for authenticated client
@pytest.fixture
async def authenticated_client(setup_database):
    # Create a test user
    user_data = {
        "email": "test@example.com",
        "password": "Password123!",
        "firstName": "Test",
        "lastName": "User"
    }
    
    # Register the user
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 201
    
    # Login to get access token
    login_data = {
        "username": user_data["email"],
        "password": user_data["password"]
    }
    response = client.post("/api/auth/token", data=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    
    # Return client with authentication headers
    client.headers = {
        "Authorization": f"Bearer {token}"
    }
    
    return client
