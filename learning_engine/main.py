"""
Learning Engine - Unified AI Coach with Real-time Chat
Handles: Session management, AI conversations, mode transitions, Socratic hints
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict
import uuid
import os
import json
import anthropic
import redis
from jose import jwt, JWTError

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/learning_companion")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"

# Database setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis for session state
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Anthropic client
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# FastAPI app
app = FastAPI(title="Learning Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Database Models =============

class LearningSession(Base):
    __tablename__ = "learning_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    goal = Column(String(50), nullable=False)  # understand, practice, review, explore
    energy_level = Column(String(20), nullable=False)  # low, medium, high
    duration_minutes = Column(Integer, nullable=False)
    input_preference = Column(String(20))  # reading, visual, mixed
    current_mode = Column(String(50), default="ask")  # ask, practice, read, watch
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    session_state = Column(JSON, default={})

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

# ============= Schemas =============

class SessionCreate(BaseModel):
    goal: str
    energy: str
    duration: int
    inputPreference: str

class SessionResponse(BaseModel):
    id: str
    goal: str
    energy_level: str
    duration_minutes: int
    current_mode: str
    created_at: datetime

class MessageCreate(BaseModel):
    content: str

# ============= Auth Helper =============

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= AI Logic =============

class LearningCoach:
    """Unified AI coach that handles conversations and mode transitions"""
    
    def __init__(self, session_id: str, session_config: dict):
        self.session_id = session_id
        self.config = session_config
        self.conversation_history = []
        
    def build_system_prompt(self) -> str:
        """Build system prompt based on session config"""
        goal_context = {
            "understand": "Focus on deep conceptual understanding. Use Socratic questioning and analogies.",
            "practice": "Provide practice problems with hints. Guide towards solutions without giving answers.",
            "review": "Be concise. Cover key points quickly. Test understanding with rapid questions.",
            "explore": "Encourage curiosity. Explore related concepts. Make connections across topics."
        }
        
        energy_context = {
            "low": "Keep explanations simple and encouraging. Shorter responses. More positive reinforcement.",
            "medium": "Balanced pace. Mix explanation with engagement.",
            "high": "Challenge with deeper questions. Longer explorations. Push boundaries."
        }
        
        return f"""You are an AI learning coach. Session context:
- Goal: {self.config['goal']} - {goal_context.get(self.config['goal'], '')}
- Energy: {self.config['energy_level']} - {energy_context.get(self.config['energy_level'], '')}
- Duration: {self.config['duration_minutes']} minutes
- Preference: {self.config.get('input_preference', 'mixed')}

Your role:
1. Adapt your teaching style to the student's goal and energy level
2. Use Socratic questioning to guide discovery
3. Provide hints rather than direct answers for practice problems
4. Encourage "return to book" when conceptual gaps appear
5. Make learning engaging and personalized

Keep responses concise but informative. Focus on understanding over memorization."""

    async def generate_response(self, user_message: str) -> dict:
        """Generate AI response using Claude"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages for Claude
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            *self.conversation_history
        ]
        
        # Call Claude API
        try:
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[msg for msg in messages if msg["role"] != "system"],
                system=self.build_system_prompt()
            )
            
            assistant_message = response.content[0].text
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Analyze if we should suggest mode transitions or hints
            response_data = {
                "role": "assistant",
                "content": assistant_message,
            }
            
            # Simple heuristics for hints (can be made smarter)
            if "?" in user_message and len(user_message.split()) < 10:
                response_data["hint"] = "Try breaking this down into smaller steps."
            
            return response_data
            
        except Exception as e:
            print(f"Claude API error: {e}")
            return {
                "role": "assistant",
                "content": "I'm having trouble connecting right now. Please try again in a moment."
            }

# ============= WebSocket Manager =============

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_coaches: Dict[str, LearningCoach] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket, session_config: dict):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_coaches[session_id] = LearningCoach(session_id, session_config)
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.session_coaches.pop(session_id, None)
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)
    
    async def handle_message(self, session_id: str, user_message: str, db: Session):
        """Process user message and generate response"""
        coach = self.session_coaches.get(session_id)
        if not coach:
            return
        
        # Save user message to DB
        user_msg = Message(
            session_id=session_id,
            role="user",
            content=user_message
        )
        db.add(user_msg)
        db.commit()
        
        # Generate AI response
        response = await coach.generate_response(user_message)
        
        # Save assistant message to DB
        assistant_msg = Message(
            session_id=session_id,
            role="assistant",
            content=response["content"],
            metadata={"hint": response.get("hint")}
        )
        db.add(assistant_msg)
        db.commit()
        
        # Send to client
        await self.send_message(session_id, response)

manager = ConnectionManager()

# ============= API Endpoints =============

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    print("✓ Learning Engine started")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/learning/session", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create a new learning session"""
    
    # Create session in DB
    session = LearningSession(
        user_id=token.get("sub"),  # User ID from JWT
        goal=session_data.goal,
        energy_level=session_data.energy,
        duration_minutes=session_data.duration,
        input_preference=session_data.inputPreference,
        current_mode="ask"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Store session config in Redis for WebSocket access
    redis_client.setex(
        f"session:{session.id}",
        3600,  # 1 hour TTL
        json.dumps({
            "goal": session.goal,
            "energy_level": session.energy_level,
            "duration_minutes": session.duration_minutes,
            "input_preference": session.input_preference
        })
    )
    
    return SessionResponse(
        id=str(session.id),
        goal=session.goal,
        energy_level=session.energy_level,
        duration_minutes=session.duration_minutes,
        current_mode=session.current_mode,
        created_at=session.created_at
    )

@app.get("/api/v1/learning/sessions")
async def get_sessions(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user's learning sessions"""
    sessions = db.query(LearningSession).filter(
        LearningSession.user_id == token.get("sub")
    ).order_by(LearningSession.created_at.desc()).limit(10).all()
    
    return [
        {
            "id": str(s.id),
            "goal": s.goal,
            "duration": s.duration_minutes,
            "date": s.created_at.strftime("%Y-%m-%d"),
            "subject": s.goal.title()
        }
        for s in sessions
    ]

@app.websocket("/api/v1/learning/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    
    # Get session config from Redis
    session_data = redis_client.get(f"session:{session_id}")
    if not session_data:
        await websocket.close(code=4000, reason="Session not found")
        return
    
    session_config = json.loads(session_data)
    
    # Connect
    await manager.connect(session_id, websocket, session_config)
    
    # Send welcome message
    await manager.send_message(session_id, {
        "role": "assistant",
        "content": f"Hi! I'm your AI learning coach. I see you want to {session_config['goal']} today. What would you like to focus on?"
    })
    
    try:
        db = next(get_db())
        while True:
            # Receive message
            data = await websocket.receive_json()
            user_message = data.get("content", "")
            
            if user_message:
                await manager.handle_message(session_id, user_message, db)
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(session_id)
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)