import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add the application directory to the Python path
sys.path.append("/app")

from app.main import app
from app.core.config import settings

# Create test client
client = TestClient(app)

# Fixture for authenticated client
@pytest.fixture
def authenticated_client():
    # Create a test user
    user_data = {
        "email": "test@example.com",
        "password": "Password123!",
        "firstName": "Test",
        "lastName": "User"
    }
    
    # Register the user
    response = client.post("/api/auth/register", json=user_data)
    
    # If user already exists, just login
    if response.status_code != 201:
        pass
    
    # Login to get access token
    login_data = {
        "username": user_data["email"],
        "password": user_data["password"]
    }
    response = client.post("/api/auth/token", data=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    
    # Return client with authentication headers
    headers = {"Authorization": f"Bearer {token}"}
    
    return TestClient(app, headers=headers)
