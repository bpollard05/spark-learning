# services/user_service/main.py
"""
User Service - Authentication, Authorization, and Profile Management
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, DECIMAL, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uuid
import os

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/learning_companion")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Database Setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=20, max_overflow=40)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# FastAPI app
app = FastAPI(title="User Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Models =============

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="student")
    email_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    display_name = Column(String(255))
    avatar_url = Column(String(500))
    grade_level = Column(String(50))
    school_id = Column(UUID(as_uuid=True))
    accessibility_needs = Column(JSON, default={})
    language_preference = Column(String(10), default="en")
    timezone = Column(String(50), default="UTC")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LearningProfile(Base):
    __tablename__ = "learning_profiles"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    modality_preferences = Column(JSON, default={"visual": 0.5, "auditory": 0.5, "kinesthetic": 0.5})
    pace_preference = Column(String(20), default="medium")
    detail_level = Column(String(20), default="medium")
    tts_rate = Column(DECIMAL(3, 2), default=1.00)
    hint_frequency = Column(String(20), default="moderate")
    preferred_explanation_style = Column(String(50))
    reading_level = Column(DECIMAL(3, 1))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============= Schemas =============

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    display_name: str
    role: str = "student"
    grade_level: str = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['student', 'teacher', 'parent', 'admin']:
            raise ValueError('Invalid role')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    display_name: str = None
    email_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class LearningProfileUpdate(BaseModel):
    modality_preferences: dict = None
    pace_preference: str = None
    detail_level: str = None
    tts_rate: float = None
    hint_frequency: str = None
    preferred_explanation_style: str = None
    reading_level: float = None

class UserProfileUpdate(BaseModel):
    display_name: str = None
    avatar_url: str = None
    grade_level: str = None
    accessibility_needs: dict = None
    language_preference: str = None
    timezone: str = None

# ============= Dependencies =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# ============= Routes =============

@app.post("/api/v1/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        role=user_data.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create user profile
    profile = UserProfile(
        user_id=user.id,
        display_name=user_data.display_name,
        grade_level=user_data.grade_level
    )
    db.add(profile)
    
    # Create learning profile with defaults
    learning_profile = LearningProfile(user_id=user.id)
    db.add(learning_profile)
    
    db.commit()
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        role=user.role,
        display_name=user_data.display_name,
        email_verified=user.email_verified,
        created_at=user.created_at
    )

@app.post("/api/v1/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return tokens"""
    
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@app.post("/api/v1/auth/refresh", response_model=Token)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new tokens
        access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
        new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return Token(access_token=access_token, refresh_token=new_refresh_token)
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate refresh token"
        )

@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user(current_user: User = Depends(verify_token), db: Session = Depends(get_db)):
    """Get current user profile"""
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        role=current_user.role,
        display_name=profile.display_name if profile else None,
        email_verified=current_user.email_verified,
        created_at=current_user.created_at
    )

@app.patch("/api/v1/users/me/profile")
async def update_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    
    # Update fields
    update_data = profile_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(profile, field, value)
    
    profile.updated_at = datetime.utcnow()
    db.commit()
    
    return {"success": True, "message": "Profile updated"}

@app.get("/api/v1/users/me/learning-profile")
async def get_learning_profile(current_user: User = Depends(verify_token), db: Session = Depends(get_db)):
    """Get user's learning profile"""
    
    profile = db.query(LearningProfile).filter(LearningProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning profile not found")
    
    return {
        "user_id": str(profile.user_id),
        "modality_preferences": profile.modality_preferences,
        "pace_preference": profile.pace_preference,
        "detail_level": profile.detail_level,
        "tts_rate": float(profile.tts_rate),
        "hint_frequency": profile.hint_frequency,
        "preferred_explanation_style": profile.preferred_explanation_style,
        "reading_level": float(profile.reading_level) if profile.reading_level else None,
        "updated_at": profile.updated_at
    }

@app.patch("/api/v1/users/me/learning-profile")
async def update_learning_profile(
    profile_data: LearningProfileUpdate,
    current_user: User = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Update user's learning profile"""
    
    profile = db.query(LearningProfile).filter(LearningProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning profile not found")
    
    # Update fields
    update_data = profile_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(profile, field, value)
    
    profile.updated_at = datetime.utcnow()
    db.commit()
    
    return {"success": True, "message": "Learning profile updated"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "user-service"}

@app.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness check - verify DB connection"""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        # Surface error message to aid debugging of readiness
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not ready: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)