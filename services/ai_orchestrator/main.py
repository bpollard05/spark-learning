# services/ai_orchestrator/main.py
"""
AI Orchestrator Service - Claude API Integration and Context Management
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import anthropic
import os
import json
import redis
from datetime import datetime, timedelta
import uuid
import hashlib

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL", "http://user-service:8001")
MAX_CONTEXT_TOKENS = 100000
CACHE_TTL = 3600  # 1 hour

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Security
security = HTTPBearer()

# FastAPI app
app = FastAPI(title="AI Orchestrator Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Prompt Templates =============

PROMPT_TEMPLATES = {
    "explain_concept": """You are an expert tutor adapting explanations to student needs.

Student Profile:
- Reading Level: {reading_level}
- Preferred Style: {explanation_style}
- Learning Pace: {pace_preference}

Topic: {topic}

{context}

Provide a clear explanation at grade {reading_level} level. Use {explanation_style} style.
Include 2-3 concrete examples that relate to everyday life.
Break down complex terms and check for understanding.
""",

    "check_math_work": """You are a patient math tutor checking student work step-by-step.

Problem: {problem}

Student's Solution:
{solution_steps}

For EACH step in the student's work:
1. Identify if the step is mathematically correct
2. If incorrect, specify the error type: arithmetic, sign error, unit mismatch, algebraic mistake, or logical gap
3. Provide a Socratic hint that guides without giving the answer
4. Explain what partial credit should be earned

Return your analysis as a JSON object with this structure:
{{
  "steps": [
    {{
      "step_number": 1,
      "student_work": "exact student text",
      "is_correct": true/false,
      "error_type": null or "arithmetic|sign|unit|algebra|logic",
      "error_explanation": "specific explanation",
      "hint": "Socratic question or guidance",
      "partial_credit": "percentage and reasoning"
    }}
  ],
  "overall_assessment": "summary of performance",
  "mastery_level": "emerging|developing|proficient|advanced",
  "next_steps": "recommended practice"
}}

Be encouraging but honest. Focus on the process, not just the answer.
""",

    "generate_hint": """You are providing a hint to guide a student who is stuck.

Problem: {problem}
Student's Current Work: {current_work}
Student's Mastery Level: {mastery_level}
Hint Level Requested: {hint_level}

Based on the mastery level, provide a {hint_level} hint:
- minimal: Just a question to redirect thinking
- moderate: Point to the relevant concept or formula
- detailed: Walk through the next logical step

Do NOT solve the problem. Guide the student to discover the solution themselves.
""",

    "summarize_content": """You are helping a student understand key concepts from their reading.

Content: {content}
Assignment Focus: {assignment_focus}
Detail Level: {detail_level}

Create a {detail_level} summary that:
1. Highlights the most important concepts related to the assignment
2. Explains why each concept matters
3. Connects ideas to make them memorable
4. Uses simple language appropriate for the student's level

Format your response clearly with sections and bullet points where helpful.
""",

    "generate_examples": """You are creating practice examples for a student.

Concept: {concept}
Current Understanding: {understanding_level}
Learning Style: {learning_style}

Generate 3 examples that:
1. Start simple and build in complexity
2. Relate to real-world scenarios the student can visualize
3. Match their learning style ({learning_style})
4. Include the worked solution for each

Make the examples engaging and relevant to a student's life.
""",

    "writing_coach": """You are a writing coach helping a student improve their essay or response.

Student's Writing:
{student_writing}

Assignment Requirements:
{requirements}

Analyze the writing for:
1. Claim/Thesis clarity and strength
2. Evidence and support quality
3. Reasoning and analysis depth
4. Organization and coherence
5. Citation accuracy (if required)

Provide constructive feedback with specific suggestions for improvement.
Do NOT rewrite the content - help the student develop their own voice.

Focus on the top 3 areas for improvement and be encouraging about strengths.
"""
}

# ============= Models =============

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class QuestionRequest(BaseModel):
    question: str
    context_ids: List[str] = []
    user_context: Optional[Dict[str, Any]] = None

class ExplainRequest(BaseModel):
    topic: str
    context: str = ""
    detail_level: str = "medium"

class CheckWorkRequest(BaseModel):
    problem: str
    solution_steps: str
    subject: str = "math"

class HintRequest(BaseModel):
    problem: str
    current_work: str
    hint_level: str = "moderate"

class SummarizeRequest(BaseModel):
    content: str
    assignment_focus: str = ""
    detail_level: str = "section"

class GenerateExamplesRequest(BaseModel):
    concept: str
    difficulty: str = "medium"

class ChatSessionRequest(BaseModel):
    session_type: str = "study_buddy"
    initial_context: Optional[Dict[str, Any]] = None

class ChatMessageRequest(BaseModel):
    message: str

class AIResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}
    cached: bool = False
    processing_time_ms: int = 0

# ============= Conversation Manager =============

class ConversationManager:
    """Manages conversation history and context for Claude API calls"""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.redis_key = f"conversation:{session_id}"
        self.max_tokens = MAX_CONTEXT_TOKENS
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Retrieve conversation history from Redis"""
        messages_json = redis_client.get(self.redis_key)
        if messages_json:
            return json.loads(messages_json)
        return []
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        messages = self.get_messages()
        messages.append({"role": role, "content": content})
        
        # Prune if needed (keep last ~50 messages or until token limit)
        if len(messages) > 50:
            messages = messages[-50:]
        
        redis_client.setex(
            self.redis_key,
            timedelta(hours=24),
            json.dumps(messages)
        )
    
    async def get_response(self, user_message: str, system_prompt: str = None) -> str:
        """Get response from Claude API with conversation context"""
        self.add_message("user", user_message)
        messages = self.get_messages()
        
        try:
            # Build request
            request_params = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": messages
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Call Claude API
            response = anthropic_client.messages.create(**request_params)
            
            assistant_message = response.content[0].text
            self.add_message("assistant", assistant_message)
            
            return assistant_message
            
        except anthropic.APIError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Claude API error: {str(e)}"
            )
    
    def clear_history(self):
        """Clear conversation history"""
        redis_client.delete(self.redis_key)

# ============= Helper Functions =============

def generate_cache_key(request_type: str, content: str) -> str:
    """Generate cache key for request"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"ai_cache:{request_type}:{content_hash}"

def get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached AI response"""
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

def cache_response(cache_key: str, response: str, ttl: int = CACHE_TTL):
    """Cache AI response"""
    redis_client.setex(cache_key, ttl, json.dumps(response))

async def call_claude_api(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: str = None
) -> str:
    """Call Claude API with retry logic"""
    
    try:
        request_params = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        response = anthropic_client.messages.create(**request_params)
        return response.content[0].text
        
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again shortly."
        )
    except anthropic.APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {str(e)}"
        )

def build_prompt_from_template(template_name: str, **kwargs) -> str:
    """Build prompt from template with provided variables"""
    template = PROMPT_TEMPLATES.get(template_name)
    if not template:
        raise ValueError(f"Template {template_name} not found")
    
    return template.format(**kwargs)

# ============= Mock User Context (Replace with actual service call) =============

async def get_user_learning_profile(user_id: str) -> Dict[str, Any]:
    """Get user's learning profile - mock for now"""
    # In production, call User Service API
    return {
        "reading_level": "10",
        "explanation_style": "conversational",
        "pace_preference": "medium",
        "modality_preferences": {"visual": 0.7, "auditory": 0.5, "kinesthetic": 0.8},
        "tts_rate": 1.0,
        "hint_frequency": "moderate"
    }

# ============= Routes =============

@app.post("/api/v1/ai/question", response_model=AIResponse)
async def answer_question(request: QuestionRequest):
    """Answer a student question with context"""
    
    start_time = datetime.now()
    
    # Check cache
    cache_key = generate_cache_key("question", request.question)
    cached_response = get_cached_response(cache_key)
    
    if cached_response:
        return AIResponse(
            response=cached_response,
            cached=True,
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
        )
    
    # Get user profile (mock for now)
    profile = await get_user_learning_profile("mock_user_id")
    
    # Build context-aware prompt
    prompt = f"""Student Question: {request.question}

Reading Level: Grade {profile['reading_level']}
Explanation Style: {profile['explanation_style']}

Provide a clear, helpful answer at the appropriate level.
Use examples and check for understanding."""
    
    # Call Claude API
    response_text = await call_claude_api(prompt)
    
    # Cache response
    cache_response(cache_key, response_text)
    
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return AIResponse(
        response=response_text,
        metadata={"reading_level": profile['reading_level']},
        processing_time_ms=processing_time
    )

@app.post("/api/v1/ai/explain", response_model=AIResponse)
async def explain_concept(request: ExplainRequest):
    """Explain a concept with adaptive complexity"""
    
    start_time = datetime.now()
    profile = await get_user_learning_profile("mock_user_id")
    
    prompt = build_prompt_from_template(
        "explain_concept",
        reading_level=profile['reading_level'],
        explanation_style=profile['explanation_style'],
        pace_preference=profile['pace_preference'],
        topic=request.topic,
        context=request.context
    )
    
    response_text = await call_claude_api(prompt)
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return AIResponse(
        response=response_text,
        processing_time_ms=processing_time
    )

@app.post("/api/v1/ai/check-work", response_model=AIResponse)
async def check_student_work(request: CheckWorkRequest):
    """Check student work and provide feedback"""
    
    start_time = datetime.now()
    profile = await get_user_learning_profile("mock_user_id")
    
    prompt = build_prompt_from_template(
        "check_math_work",
        problem=request.problem,
        solution_steps=request.solution_steps
    )
    
    response_text = await call_claude_api(
        prompt,
        temperature=0.3  # Lower temperature for analytical tasks
    )
    
    # Try to parse JSON response
    try:
        feedback = json.loads(response_text)
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return AIResponse(
            response=response_text,
            metadata={
                "structured": True,
                "feedback": feedback
            },
            processing_time_ms=processing_time
        )
    except json.JSONDecodeError:
        # Return as plain text if JSON parsing fails
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        return AIResponse(
            response=response_text,
            metadata={"structured": False},
            processing_time_ms=processing_time
        )

@app.post("/api/v1/ai/generate-hint", response_model=AIResponse)
async def generate_hint(request: HintRequest):
    """Generate a hint for a stuck student"""
    
    start_time = datetime.now()
    profile = await get_user_learning_profile("mock_user_id")
    
    # Determine mastery level based on profile (simplified)
    mastery_level = "developing"  # Would come from analytics in production
    
    prompt = build_prompt_from_template(
        "generate_hint",
        problem=request.problem,
        current_work=request.current_work,
        mastery_level=mastery_level,
        hint_level=request.hint_level
    )
    
    response_text = await call_claude_api(prompt)
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return AIResponse(
        response=response_text,
        metadata={"hint_level": request.hint_level},
        processing_time_ms=processing_time
    )

@app.post("/api/v1/ai/summarize", response_model=AIResponse)
async def summarize_content(request: SummarizeRequest):
    """Summarize content at specified detail level"""
    
    start_time = datetime.now()
    
    # Check cache
    cache_key = generate_cache_key("summarize", request.content + request.detail_level)
    cached_response = get_cached_response(cache_key)
    
    if cached_response:
        return AIResponse(
            response=cached_response,
            cached=True,
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
        )
    
    prompt = build_prompt_from_template(
        "summarize_content",
        content=request.content,
        assignment_focus=request.assignment_focus,
        detail_level=request.detail_level
    )
    
    response_text = await call_claude_api(prompt, max_tokens=3000)
    
    # Cache summary
    cache_response(cache_key, response_text, ttl=7200)  # 2 hours
    
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return AIResponse(
        response=response_text,
        metadata={"detail_level": request.detail_level},
        processing_time_ms=processing_time
    )

@app.post("/api/v1/ai/generate-examples", response_model=AIResponse)
async def generate_examples(request: GenerateExamplesRequest):
    """Generate practice examples for a concept"""
    
    start_time = datetime.now()
    profile = await get_user_learning_profile("mock_user_id")
    
    # Determine learning style from modality preferences
    learning_style = "visual" if profile['modality_preferences']['visual'] > 0.6 else "mixed"
    
    prompt = build_prompt_from_template(
        "generate_examples",
        concept=request.concept,
        understanding_level=request.difficulty,
        learning_style=learning_style
    )
    
    response_text = await call_claude_api(prompt, max_tokens=3000)
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return AIResponse(
        response=response_text,
        metadata={"difficulty": request.difficulty, "learning_style": learning_style},
        processing_time_ms=processing_time
    )

# ============= Chat Session Routes =============

@app.post("/api/v1/ai/chat/session")
async def create_chat_session(request: ChatSessionRequest):
    """Create a new chat session"""
    
    session_id = str(uuid.uuid4())
    user_id = "mock_user_id"  # Would come from auth
    
    # Initialize conversation manager
    conv_manager = ConversationManager(session_id, user_id)
    
    # Add initial context if provided
    if request.initial_context:
        context_message = f"Context: {json.dumps(request.initial_context)}"
        conv_manager.add_message("user", context_message)
    
    return {
        "session_id": session_id,
        "session_type": request.session_type,
        "created_at": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/ai/chat/session/{session_id}/message", response_model=AIResponse)
async def send_chat_message(session_id: str, request: ChatMessageRequest):
    """Send a message in a chat session"""
    
    start_time = datetime.now()
    user_id = "mock_user_id"  # Would come from auth
    
    conv_manager = ConversationManager(session_id, user_id)
    
    # Get user profile for personalization
    profile = await get_user_learning_profile(user_id)
    
    system_prompt = f"""You are a helpful AI tutor having a conversation with a student.
    
Student Profile:
- Reading Level: Grade {profile['reading_level']}
- Learning Pace: {profile['pace_preference']}
- Preferred Style: {profile['explanation_style']}

Be encouraging, patient, and adaptive. Use Socratic questioning to guide learning.
Break down complex concepts. Check for understanding regularly."""
    
    # Get response
    response_text = await conv_manager.get_response(request.message, system_prompt)
    
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return AIResponse(
        response=response_text,
        metadata={"session_id": session_id},
        processing_time_ms=processing_time
    )

@app.get("/api/v1/ai/chat/session/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    
    user_id = "mock_user_id"  # Would come from auth
    conv_manager = ConversationManager(session_id, user_id)
    messages = conv_manager.get_messages()
    
    return {
        "session_id": session_id,
        "messages": messages,
        "message_count": len(messages)
    }

@app.delete("/api/v1/ai/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    
    user_id = "mock_user_id"  # Would come from auth
    conv_manager = ConversationManager(session_id, user_id)
    conv_manager.clear_history()
    
    return {"success": True, "message": "Session deleted"}

# ============= Health Checks =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-orchestrator"}

@app.get("/ready")
async def readiness_check():
    """Readiness check - verify dependencies"""
    try:
        # Check Redis
        redis_client.ping()
        
        # Check Claude API (lightweight call)
        anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)