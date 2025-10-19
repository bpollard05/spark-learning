# tests/conftest.py
"""
Pytest configuration and fixtures
"""
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Test database URL
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/test_db")

@pytest.fixture(scope="session")
def engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL)
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def db_session(engine):
    """Create database session for each test"""
    from services.user_service.main import Base
    
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    """Create test client"""
    from services.user_service.main import app
    return TestClient(app)

@pytest.fixture
def auth_token(client):
    """Get authentication token"""
    # Register user
    response = client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "TestPassword123",
        "display_name": "Test User",
        "role": "student"
    })
    
    # Login
    response = client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "TestPassword123"
    })
    
    return response.json()["access_token"]

@pytest.fixture
def auth_headers(auth_token):
    """Get authentication headers"""
    return {"Authorization": f"Bearer {auth_token}"}

# =====================================================
# tests/test_user_service.py
# =====================================================
"""
Tests for User Service
"""
import pytest
from fastapi.testclient import TestClient

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_user_registration(client):
    """Test user registration"""
    response = client.post("/api/v1/auth/register", json={
        "email": "newuser@example.com",
        "password": "SecurePass123",
        "display_name": "New User",
        "role": "student",
        "grade_level": "Grade 10"
    })
    
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["role"] == "student"
    assert "id" in data

def test_duplicate_registration(client):
    """Test duplicate email registration fails"""
    # Register first user
    client.post("/api/v1/auth/register", json={
        "email": "duplicate@example.com",
        "password": "Pass123",
        "display_name": "User 1",
        "role": "student"
    })
    
    # Try to register again with same email
    response = client.post("/api/v1/auth/register", json={
        "email": "duplicate@example.com",
        "password": "Pass456",
        "display_name": "User 2",
        "role": "student"
    })
    
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()

def test_login(client):
    """Test user login"""
    # Register user
    client.post("/api/v1/auth/register", json={
        "email": "login@example.com",
        "password": "LoginPass123",
        "display_name": "Login User",
        "role": "student"
    })
    
    # Login
    response = client.post("/api/v1/auth/login", json={
        "email": "login@example.com",
        "password": "LoginPass123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_credentials(client):
    """Test login with invalid credentials"""
    response = client.post("/api/v1/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "WrongPassword"
    })
    
    assert response.status_code == 401

def test_get_current_user(client, auth_headers):
    """Test getting current user profile"""
    response = client.get("/api/v1/users/me", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert "role" in data

def test_update_profile(client, auth_headers):
    """Test updating user profile"""
    response = client.patch("/api/v1/users/me/profile", 
        headers=auth_headers,
        json={
            "display_name": "Updated Name",
            "grade_level": "Grade 11"
        }
    )
    
    assert response.status_code == 200

def test_get_learning_profile(client, auth_headers):
    """Test getting learning profile"""
    response = client.get("/api/v1/users/me/learning-profile", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "modality_preferences" in data
    assert "pace_preference" in data

# =====================================================
# tests/test_ai_orchestrator.py
# =====================================================
"""
Tests for AI Orchestrator Service
"""
import pytest

def test_ask_question(client):
    """Test asking AI a question"""
    response = client.post("/api/v1/ai/question", json={
        "question": "What is 2 + 2?",
        "context_ids": []
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["processing_time_ms"] > 0

def test_explain_concept(client):
    """Test explaining a concept"""
    response = client.post("/api/v1/ai/explain", json={
        "topic": "photosynthesis",
        "context": "",
        "detail_level": "medium"
    })
    
    assert response.status_code == 200
    assert "response" in response.json()

def test_check_work(client):
    """Test checking student work"""
    response = client.post("/api/v1/ai/check-work", json={
        "problem": "Solve: x + 5 = 10",
        "solution_steps": "x = 5",
        "subject": "math"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data

def test_generate_hint(client):
    """Test generating a hint"""
    response = client.post("/api/v1/ai/generate-hint", json={
        "problem": "Find the derivative of x^2",
        "current_work": "I'm not sure where to start",
        "hint_level": "moderate"
    })
    
    assert response.status_code == 200
    assert "response" in response.json()

def test_create_chat_session(client):
    """Test creating a chat session"""
    response = client.post("/api/v1/ai/chat/session", json={
        "session_type": "study_buddy"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data

def test_send_chat_message(client):
    """Test sending a chat message"""
    # Create session
    session_response = client.post("/api/v1/ai/chat/session", json={
        "session_type": "study_buddy"
    })
    session_id = session_response.json()["session_id"]
    
    # Send message
    response = client.post(f"/api/v1/ai/chat/session/{session_id}/message", json={
        "message": "Help me understand quadratic equations"
    })
    
    assert response.status_code == 200
    assert "response" in response.json()

# =====================================================
# tests/test_gamification.py
# =====================================================
"""
Tests for Gamification Service
"""
import pytest
import uuid

def test_get_progress(client):
    """Test getting user progress"""
    user_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/progress/{user_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "total_xp" in data
    assert "level" in data
    assert data["level"] >= 1

def test_award_xp(client):
    """Test awarding XP"""
    user_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/progress/{user_id}/award-xp", json={
        "activity_type": "practice_problem",
        "context": {
            "difficulty": 1.5,
            "accuracy": 0.9,
            "attempts": 1
        }
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["xp_earned"] > 0
    assert "level" in data

def test_get_achievements(client):
    """Test getting achievements"""
    user_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/progress/{user_id}/achievements")
    
    assert response.status_code == 200
    data = response.json()
    assert "achievements" in data

def test_get_roadmap(client):
    """Test getting learning roadmap"""
    user_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/progress/{user_id}/roadmap")
    
    assert response.status_code == 200
    data = response.json()
    assert "roadmap" in data

def test_leaderboard(client):
    """Test getting leaderboard"""
    response = client.get("/api/v1/leaderboard", params={
        "metric": "xp",
        "limit": 10
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "leaderboard" in data

# =====================================================
# tests/test_integration.py
# =====================================================
"""
Integration tests - full user flows
"""
import pytest

def test_student_onboarding_flow(client):
    """Test complete student onboarding flow"""
    
    # Step 1: Register
    register_response = client.post("/api/v1/auth/register", json={
        "email": "integration@example.com",
        "password": "IntegrationTest123",
        "display_name": "Integration Test",
        "role": "student",
        "grade_level": "Grade 10"
    })
    assert register_response.status_code == 201
    user_id = register_response.json()["id"]
    
    # Step 2: Login
    login_response = client.post("/api/v1/auth/login", json={
        "email": "integration@example.com",
        "password": "IntegrationTest123"
    })
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Step 3: Get initial progress
    progress_response = client.get(f"/api/v1/progress/{user_id}")
    assert progress_response.status_code == 200
    initial_xp = progress_response.json()["total_xp"]
    
    # Step 4: Complete an activity and earn XP
    xp_response = client.post(f"/api/v1/progress/{user_id}/award-xp", json={
        "activity_type": "practice_problem",
        "context": {"difficulty": 1.0, "accuracy": 1.0}
    })
    assert xp_response.status_code == 200
    xp_earned = xp_response.json()["xp_earned"]
    assert xp_earned > 0
    
    # Step 5: Verify progress updated
    final_progress = client.get(f"/api/v1/progress/{user_id}")
    assert final_progress.json()["total_xp"] == initial_xp + xp_earned

def test_learning_session_flow(client):
    """Test complete learning session flow"""
    
    # Create chat session
    session_response = client.post("/api/v1/ai/chat/session", json={
        "session_type": "study_buddy",
        "initial_context": {"topic": "algebra"}
    })
    session_id = session_response.json()["session_id"]
    
    # Ask question
    msg1 = client.post(f"/api/v1/ai/chat/session/{session_id}/message", json={
        "message": "How do I solve x^2 + 5x + 6 = 0?"
    })
    assert msg1.status_code == 200
    assert len(msg1.json()["response"]) > 0
    
    # Ask follow-up
    msg2 = client.post(f"/api/v1/ai/chat/session/{session_id}/message", json={
        "message": "Can you explain factoring?"
    })
    assert msg2.status_code == 200
    
    # Get history
    history = client.get(f"/api/v1/ai/chat/session/{session_id}/history")
    assert history.status_code == 200
    assert len(history.json()["messages"]) >= 4  # 2 user + 2 assistant

# =====================================================
# tests/test_performance.py
# =====================================================
"""
Performance and load tests
"""
import pytest
import time
from concurrent.futures import ThreadPoolExecutor

def test_ai_response_time(client):
    """Test AI response time is under threshold"""
    start = time.time()
    
    response = client.post("/api/v1/ai/question", json={
        "question": "What is the capital of France?"
    })
    
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 5.0  # Should respond within 5 seconds

def test_concurrent_requests(client):
    """Test handling concurrent requests"""
    
    def make_request():
        return client.get("/health")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        responses = [f.result() for f in futures]
    
    # All requests should succeed
    assert all(r.status_code == 200 for r in responses)

def test_database_query_performance(client, auth_headers):
    """Test database query performance"""
    start = time.time()
    
    response = client.get("/api/v1/users/me", headers=auth_headers)
    
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 0.5  # Database queries should be fast

# =====================================================
# Run tests with: pytest -v tests/
# Generate coverage report: pytest --cov=services --cov-report=html
# =====================================================