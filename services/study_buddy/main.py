# services/study_buddy/main.py
"""
Study Buddy Service - Interactive AI Tutoring with Error Detection
"""
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import anthropic
import os
import json
import uuid
from datetime import datetime
import re

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/learning_companion")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Study Buddy Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Database Models =============

class StudySession(Base):
    __tablename__ = "study_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True))
    assignment_id = Column(UUID(as_uuid=True))
    session_type = Column(String(50), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    total_duration_seconds = Column(Integer)
    topic = Column(String(255))
    session_metadata = Column(JSON, default={})

class AIInteraction(Base):
    __tablename__ = "ai_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    interaction_type = Column(String(50), nullable=False)
    user_input = Column(Text)
    ai_response = Column(Text)
    response_metadata = Column(JSON)
    was_helpful = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ErrorLog(Base):
    __tablename__ = "error_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True))
    problem_id = Column(UUID(as_uuid=True))
    error_type = Column(String(100), nullable=False)
    error_details = Column(JSON)
    correction_provided = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ============= Pydantic Models =============

class WalkthroughRequest(BaseModel):
    problem: str
    subject: str
    difficulty: str = "medium"
    show_solution: bool = False

class ErrorDetectionRequest(BaseModel):
    problem: str
    student_solution: str
    step_by_step: bool = True

class SocraticQuestionRequest(BaseModel):
    topic: str
    student_response: str
    depth_level: int = Field(1, ge=1, le=5)

class WritingFeedbackRequest(BaseModel):
    essay: str
    rubric: Optional[Dict[str, Any]] = None
    focus_areas: List[str] = []

class Step(BaseModel):
    step_number: int
    description: str
    explanation: str
    is_key_concept: bool = False

class WalkthroughResponse(BaseModel):
    problem: str
    steps: List[Step]
    key_concepts: List[str]
    common_mistakes: List[str]
    practice_suggestion: str

class ErrorFeedback(BaseModel):
    step_number: int
    student_work: str
    is_correct: bool
    error_type: Optional[str]
    error_explanation: Optional[str]
    hint: str
    partial_credit: str

class ErrorDetectionResponse(BaseModel):
    overall_correct: bool
    steps: List[ErrorFeedback]
    mastery_level: str
    overall_assessment: str
    next_steps: str

# ============= Dependencies =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= Helper Functions =============

def parse_math_steps(solution_text: str) -> List[str]:
    """Parse solution into individual steps"""
    # Split by common step indicators
    patterns = [
        r'\n\d+\.',  # Numbered steps
        r'\nStep \d+:',  # "Step 1:"
        r'\n-',  # Bullet points
        r'\n\n',  # Double newlines
    ]
    
    steps = []
    current = solution_text
    
    for pattern in patterns:
        if re.search(pattern, current):
            parts = re.split(pattern, current)
            steps = [p.strip() for p in parts if p.strip()]
            break
    
    # If no patterns matched, treat as single step
    if not steps:
        steps = [solution_text.strip()]
    
    return steps

def determine_mastery_level(error_count: int, total_steps: int) -> str:
    """Determine mastery level from error rate"""
    if total_steps == 0:
        return "emerging"
    
    accuracy = 1 - (error_count / total_steps)
    
    if accuracy >= 0.9:
        return "advanced"
    elif accuracy >= 0.75:
        return "proficient"
    elif accuracy >= 0.5:
        return "developing"
    else:
        return "emerging"

async def call_claude_for_tutoring(prompt: str, temperature: float = 0.7) -> str:
    """Call Claude API for tutoring interactions"""
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

# ============= Routes =============

@app.post("/api/v1/study-buddy/walkthrough", response_model=WalkthroughResponse)
async def problem_walkthrough(request: WalkthroughRequest, db: Session = Depends(get_db)):
    """Provide step-by-step walkthrough of a problem"""
    
    user_id = uuid.uuid4()  # Mock - would come from auth
    
    # Create study session
    session = StudySession(
        user_id=user_id,
        session_type="walkthrough",
        topic=request.subject
    )
    db.add(session)
    db.commit()
    
    prompt = f"""You are an expert tutor providing a step-by-step walkthrough.

Problem: {request.problem}
Subject: {request.subject}
Difficulty: {request.difficulty}

Provide a complete walkthrough with:
1. Each step clearly numbered and explained
2. Why each step is necessary (the reasoning)
3. Key concepts highlighted
4. Common mistakes students make on this type of problem
5. A suggestion for similar practice

{"Include the full worked solution." if request.show_solution else "Guide the student through the process without giving away the final answer."}

Return your response as JSON:
{{
  "steps": [
    {{
      "step_number": 1,
      "description": "Clear description of what to do",
      "explanation": "Why this step matters",
      "is_key_concept": true/false
    }}
  ],
  "key_concepts": ["concept 1", "concept 2"],
  "common_mistakes": ["mistake 1", "mistake 2"],
  "practice_suggestion": "Try a similar problem with..."
}}
"""
    
    response_text = await call_claude_for_tutoring(prompt, temperature=0.5)
    
    # Log interaction
    interaction = AIInteraction(
        session_id=session.id,
        user_id=user_id,
        interaction_type="walkthrough",
        user_input=request.problem,
        ai_response=response_text
    )
    db.add(interaction)
    db.commit()
    
    # Parse response
    try:
        result = json.loads(response_text)
        return WalkthroughResponse(**result, problem=request.problem)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return WalkthroughResponse(
            problem=request.problem,
            steps=[Step(step_number=1, description="See detailed explanation", explanation=response_text)],
            key_concepts=[],
            common_mistakes=[],
            practice_suggestion="Practice similar problems"
        )

@app.post("/api/v1/study-buddy/check-work", response_model=ErrorDetectionResponse)
async def detect_errors(request: ErrorDetectionRequest, db: Session = Depends(get_db)):
    """Detect errors in student work and provide feedback"""
    
    user_id = uuid.uuid4()  # Mock - would come from auth
    
    # Create study session
    session = StudySession(
        user_id=user_id,
        session_type="error_detection",
        topic="problem_solving"
    )
    db.add(session)
    db.commit()
    
    # Parse student solution into steps if requested
    steps_text = request.student_solution
    if request.step_by_step:
        steps_list = parse_math_steps(request.student_solution)
        steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps_list)])
    
    prompt = f"""You are a patient tutor checking a student's work for errors.

Problem: {request.problem}

Student's Solution:
{steps_text}

Analyze EACH step carefully:
1. Is the step mathematically/logically correct?
2. If incorrect, what type of error? (arithmetic, sign error, unit mismatch, algebraic mistake, logic gap)
3. Provide a Socratic hint - guide without giving the answer
4. Assess partial credit earned

Return JSON:
{{
  "steps": [
    {{
      "step_number": 1,
      "student_work": "exact text from student",
      "is_correct": true/false,
      "error_type": null or "arithmetic|sign|unit|algebra|logic",
      "error_explanation": "specific issue",
      "hint": "guiding question",
      "partial_credit": "percentage and reason"
    }}
  ],
  "overall_assessment": "encouraging summary",
  "mastery_level": "emerging|developing|proficient|advanced",
  "next_steps": "recommended practice"
}}

Be encouraging. Focus on growth and learning from mistakes.
"""
    
    response_text = await call_claude_for_tutoring(prompt, temperature=0.3)
    
    # Log interaction
    interaction = AIInteraction(
        session_id=session.id,
        user_id=user_id,
        interaction_type="error_detection",
        user_input=request.student_solution,
        ai_response=response_text
    )
    db.add(interaction)
    db.commit()
    
    # Parse response
    try:
        result = json.loads(response_text)
        
        # Log errors to error_log table
        error_count = 0
        for step in result.get('steps', []):
            if not step['is_correct']:
                error_count += 1
                error_log_entry = ErrorLog(
                    user_id=user_id,
                    session_id=session.id,
                    error_type=step.get('error_type', 'unknown'),
                    error_details={
                        "step": step['step_number'],
                        "student_work": step['student_work'],
                        "explanation": step.get('error_explanation')
                    }
                )
                db.add(error_log_entry)
        
        db.commit()
        
        # Build response
        steps = [ErrorFeedback(**step) for step in result['steps']]
        overall_correct = error_count == 0
        
        return ErrorDetectionResponse(
            overall_correct=overall_correct,
            steps=steps,
            mastery_level=result.get('mastery_level', 'developing'),
            overall_assessment=result.get('overall_assessment', ''),
            next_steps=result.get('next_steps', '')
        )
        
    except json.JSONDecodeError:
        # Fallback response
        return ErrorDetectionResponse(
            overall_correct=False,
            steps=[],
            mastery_level="unknown",
            overall_assessment=response_text,
            next_steps="Continue practicing"
        )

@app.post("/api/v1/study-buddy/socratic-question")
async def generate_socratic_question(request: SocraticQuestionRequest, db: Session = Depends(get_db)):
    """Generate Socratic questions to guide learning"""
    
    user_id = uuid.uuid4()  # Mock
    
    session = StudySession(
        user_id=user_id,
        session_type="socratic_dialogue",
        topic=request.topic
    )
    db.add(session)
    db.commit()
    
    prompt = f"""You are a Socratic tutor engaging a student in deep learning.

Topic: {request.topic}
Student's Current Understanding: {request.student_response}
Question Depth: {request.depth_level}/5 (1=surface, 5=deep)

Generate a thoughtful Socratic question that:
1. Builds on what the student said
2. Challenges them to think deeper
3. Guides toward important insights
4. Doesn't give away answers

Make the question thought-provoking but accessible.
"""
    
    response_text = await call_claude_for_tutoring(prompt, temperature=0.8)
    
    interaction = AIInteraction(
        session_id=session.id,
        user_id=user_id,
        interaction_type="socratic_question",
        user_input=request.student_response,
        ai_response=response_text,
        response_metadata={"depth_level": request.depth_level}
    )
    db.add(interaction)
    db.commit()
    
    return {
        "question": response_text,
        "depth_level": request.depth_level,
        "session_id": str(session.id)
    }

@app.post("/api/v1/study-buddy/writing-feedback")
async def provide_writing_feedback(request: WritingFeedbackRequest, db: Session = Depends(get_db)):
    """Provide feedback on student writing"""
    
    user_id = uuid.uuid4()  # Mock
    
    session = StudySession(
        user_id=user_id,
        session_type="writing_coaching",
        topic="writing"
    )
    db.add(session)
    db.commit()
    
    rubric_text = json.dumps(request.rubric, indent=2) if request.rubric else "General essay rubric"
    focus_areas_text = ", ".join(request.focus_areas) if request.focus_areas else "all aspects"
    
    prompt = f"""You are a writing coach providing constructive feedback.

Student's Writing:
{request.essay}

Rubric/Requirements:
{rubric_text}

Focus Areas: {focus_areas_text}

Analyze and provide feedback on:
1. Claim/Thesis - Is it clear, specific, and arguable?
2. Evidence - Is it relevant, sufficient, and well-integrated?
3. Reasoning - Are arguments logical and well-developed?
4. Organization - Is the structure clear and effective?
5. Citation - Are sources properly cited? (if applicable)
6. Style - Is the writing clear, engaging, appropriate?

For each area:
- Identify 1-2 specific strengths
- Suggest 1-2 concrete improvements
- Provide an example revision

Be encouraging. Help the student develop their own voice - don't rewrite for them.

Return as JSON:
{{
  "strengths": ["strength 1", "strength 2"],
  "areas_for_improvement": [
    {{
      "area": "thesis",
      "issue": "description",
      "suggestion": "specific guidance",
      "example": "revised version"
    }}
  ],
  "overall_score": "estimate based on rubric",
  "encouragement": "positive, motivating message"
}}
"""
    
    response_text = await call_claude_for_tutoring(prompt, temperature=0.6)
    
    interaction = AIInteraction(
        session_id=session.id,
        user_id=user_id,
        interaction_type="writing_feedback",
        user_input=request.essay[:500],  # Store first 500 chars
        ai_response=response_text
    )
    db.add(interaction)
    db.commit()
    
    return {"feedback": response_text, "session_id": str(session.id)}

@app.post("/api/v1/study-buddy/session")
async def create_session(
    session_type: str,
    topic: str = None,
    db: Session = Depends(get_db)
):
    """Create a new study session"""
    
    user_id = uuid.uuid4()  # Mock
    
    session = StudySession(
        user_id=user_id,
        session_type=session_type,
        topic=topic
    )
    db.add(session)
    db.commit()
    
    return {
        "session_id": str(session.id),
        "session_type": session_type,
        "started_at": session.started_at.isoformat()
    }

@app.post("/api/v1/study-buddy/session/{session_id}/end")
async def end_session(session_id: str, db: Session = Depends(get_db)):
    """End a study session"""
    
    session = db.query(StudySession).filter(StudySession.id == uuid.UUID(session_id)).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.ended_at:
        raise HTTPException(status_code=400, detail="Session already ended")
    
    session.ended_at = datetime.utcnow()
    session.total_duration_seconds = int((session.ended_at - session.started_at).total_seconds())
    db.commit()
    
    return {
        "session_id": str(session.id),
        "duration_seconds": session.total_duration_seconds,
        "ended_at": session.ended_at.isoformat()
    }

@app.get("/api/v1/study-buddy/session/{session_id}/interactions")
async def get_session_interactions(session_id: str, db: Session = Depends(get_db)):
    """Get all interactions in a session"""
    
    interactions = db.query(AIInteraction)\
        .filter(AIInteraction.session_id == uuid.UUID(session_id))\
        .order_by(AIInteraction.timestamp)\
        .all()
    
    return {
        "session_id": session_id,
        "interactions": [
            {
                "id": str(i.id),
                "type": i.interaction_type,
                "user_input": i.user_input,
                "ai_response": i.ai_response,
                "timestamp": i.timestamp.isoformat(),
                "was_helpful": i.was_helpful
            }
            for i in interactions
        ]
    }

@app.post("/api/v1/study-buddy/interaction/{interaction_id}/feedback")
async def rate_interaction(
    interaction_id: str,
    was_helpful: bool,
    db: Session = Depends(get_db)
):
    """Rate whether an AI interaction was helpful"""
    
    interaction = db.query(AIInteraction)\
        .filter(AIInteraction.id == uuid.UUID(interaction_id))\
        .first()
    
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")
    
    interaction.was_helpful = was_helpful
    db.commit()
    
    return {"success": True, "interaction_id": interaction_id}

@app.get("/api/v1/study-buddy/errors/patterns")
async def get_error_patterns(user_id: str = None, db: Session = Depends(get_db)):
    """Get common error patterns for a user"""
    
    if not user_id:
        user_id = str(uuid.uuid4())  # Mock
    
    # Query error log for patterns
    errors = db.query(ErrorLog)\
        .filter(ErrorLog.user_id == uuid.UUID(user_id))\
        .order_by(ErrorLog.timestamp.desc())\
        .limit(50)\
        .all()
    
    # Analyze patterns
    error_types = {}
    for error in errors:
        error_type = error.error_type
        if error_type in error_types:
            error_types[error_type] += 1
        else:
            error_types[error_type] = 1
    
    # Sort by frequency
    patterns = [
        {"error_type": k, "frequency": v}
        for k, v in sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        "user_id": user_id,
        "total_errors": len(errors),
        "patterns": patterns,
        "recommendation": "Focus on practice for most common error types"
    }

# ============= WebSocket for Real-Time Tutoring =============

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/study-buddy/{session_id}")
async def websocket_tutoring(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time tutoring"""
    
    await manager.connect(session_id, websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            content = data.get("content")
            
            if message_type == "question":
                # Process question and send response
                prompt = f"""Student question: {content}
                
Provide a helpful, encouraging response. Ask guiding questions if appropriate."""
                
                response = await call_claude_for_tutoring(prompt)
                
                await manager.send_message(session_id, {
                    "type": "answer",
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif message_type == "check_step":
                # Check a single step in real-time
                problem = data.get("problem")
                step = content
                
                prompt = f"""Problem: {problem}
Student's current step: {step}

Is this step correct? If not, what's the error? Provide a brief hint.

Be concise - this is real-time feedback."""
                
                response = await call_claude_for_tutoring(prompt, temperature=0.3)
                
                await manager.send_message(session_id, {
                    "type": "step_feedback",
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "study-buddy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)