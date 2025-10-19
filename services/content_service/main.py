# services/content_service/main.py
"""
Content Service - Document Upload, OCR, Segmentation, and Processing
"""
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
import os
from datetime import datetime
import boto3
from io import BytesIO
import anthropic
import json
import re
from enum import Enum

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/learning_companion")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "learning-companion-content")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Database Setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Content Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Database Models =============

class Content(Base):
    __tablename__ = "content"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    content_type = Column(String(50), nullable=False)
    storage_url = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer)
    processing_status = Column(String(50), default="pending")
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ContentSegment(Base):
    __tablename__ = "content_segments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    segment_type = Column(String(50), nullable=False)
    text_content = Column(Text)
    importance_level = Column(String(50))
    position_order = Column(Integer, nullable=False)
    parent_segment_id = Column(UUID(as_uuid=True))
    segment_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

class ContentMetadata(Base):
    __tablename__ = "content_metadata"
    
    content_id = Column(UUID(as_uuid=True), primary_key=True)
    key_terms = Column(JSON, default=[])
    reading_level = Column(Float)
    estimated_duration_mins = Column(Integer)
    summary = Column(Text)
    outline = Column(JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)

# ============= Pydantic Models =============

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ImportanceLevel(str, Enum):
    CORE = "core"
    SUPPORTING = "supporting"
    BACKGROUND = "background"

class SegmentType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    TABLE = "table"
    LIST = "list"
    CODE = "code"

class ContentUploadResponse(BaseModel):
    content_id: str
    title: str
    storage_url: str
    processing_status: str
    created_at: str

class SegmentResponse(BaseModel):
    id: str
    segment_type: str
    text_content: str
    importance_level: Optional[str]
    position_order: int

class ContentSummary(BaseModel):
    content_id: str
    summary: str
    key_terms: List[str]
    reading_level: Optional[float]
    estimated_duration_mins: Optional[int]

class TagForAssignmentRequest(BaseModel):
    assignment_focus: str
    rubric: Optional[Dict[str, Any]] = None

# ============= Dependencies =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= Helper Functions =============

def upload_to_s3(file_content: bytes, filename: str, content_type: str) -> str:
    """Upload file to S3 and return URL"""
    key = f"{uuid.uuid4()}/{filename}"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=file_content,
            ContentType=content_type
        )
        
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        return url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF using OCR (mock implementation)"""
    # In production, would use PyPDF2, pdfplumber, or Google Vision API
    # For now, return mock text
    return "Mock extracted text from PDF. This would contain the actual PDF content."

def segment_text(text: str) -> List[Dict[str, Any]]:
    """Segment text into logical units"""
    segments = []
    
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        # Detect if it's a heading (short, potentially all caps or title case)
        is_heading = len(para) < 100 and (para.isupper() or para.istitle())
        
        segment = {
            "segment_type": SegmentType.HEADING if is_heading else SegmentType.PARAGRAPH,
            "text_content": para,
            "position_order": idx,
            "metadata": {"word_count": len(para.split())}
        }
        
        segments.append(segment)
    
    return segments

async def classify_importance_with_ai(
    segments: List[Dict[str, Any]],
    assignment_focus: str = ""
) -> List[Dict[str, Any]]:
    """Use Claude to classify segment importance"""
    
    # Build prompt
    segments_text = "\n\n".join([
        f"Segment {s['position_order']}: {s['text_content'][:200]}"
        for s in segments[:20]  # Limit to first 20 for context
    ])
    
    focus_text = f"\nAssignment Focus: {assignment_focus}" if assignment_focus else ""
    
    prompt = f"""Analyze these text segments and classify each as Core, Supporting, or Background.

{segments_text}{focus_text}

Core: Essential information directly relevant to understanding the main concepts
Supporting: Important details that enhance understanding
Background: Context, examples, or tangential information

Return JSON array:
[
  {{"segment": 0, "importance": "core|supporting|background", "reason": "brief explanation"}},
  ...
]
"""
    
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            classifications = json.loads(json_match.group())
            
            # Apply classifications
            for item in classifications:
                seg_idx = item['segment']
                if seg_idx < len(segments):
                    segments[seg_idx]['importance_level'] = item['importance']
                    segments[seg_idx]['metadata']['classification_reason'] = item['reason']
        
    except Exception as e:
        print(f"AI classification failed: {e}")
        # Fallback: simple heuristic
        for segment in segments:
            if segment['segment_type'] == SegmentType.HEADING:
                segment['importance_level'] = ImportanceLevel.CORE
            else:
                segment['importance_level'] = ImportanceLevel.SUPPORTING
    
    return segments

async def generate_summary(text: str, level: str = "section") -> str:
    """Generate summary at specified detail level"""
    
    detail_prompts = {
        "overview": "Provide a 2-3 sentence overview of the main idea.",
        "section": "Provide a paragraph summarizing the key points and their relationships.",
        "detailed": "Provide a comprehensive summary covering all major concepts and supporting details."
    }
    
    prompt = f"""Summarize this content. {detail_prompts.get(level, detail_prompts['section'])}

Content:
{text[:4000]}  # Limit to avoid token overflow

Summary:"""
    
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

async def extract_key_terms(text: str) -> List[str]:
    """Extract key terms and concepts"""
    
    prompt = f"""Extract the 10-15 most important terms and concepts from this text.
Include technical terms, key concepts, and important proper nouns.

Text:
{text[:4000]}

Return as a JSON array of strings: ["term1", "term2", ...]
"""
    
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Extract JSON array
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            terms = json.loads(json_match.group())
            return terms[:15]
        
    except Exception as e:
        print(f"Key term extraction failed: {e}")
    
    return []

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes"""
    word_count = len(text.split())
    return max(1, word_count // words_per_minute)

# ============= Background Processing =============

async def process_content_background(content_id: str, file_content: bytes, content_type: str, db: Session):
    """Background task to process uploaded content"""
    
    content = db.query(Content).filter(Content.id == uuid.UUID(content_id)).first()
    if not content:
        return
    
    try:
        content.processing_status = ProcessingStatus.PROCESSING
        db.commit()
        
        # Step 1: Extract text
        if content_type == "application/pdf":
            text = extract_text_from_pdf(file_content)
        else:
            text = file_content.decode('utf-8')
        
        # Step 2: Segment text
        segments = segment_text(text)
        
        # Step 3: Classify importance
        segments = await classify_importance_with_ai(segments)
        
        # Step 4: Store segments
        for segment_data in segments:
            segment = ContentSegment(
                content_id=uuid.UUID(content_id),
                segment_type=segment_data['segment_type'],
                text_content=segment_data['text_content'],
                importance_level=segment_data.get('importance_level'),
                position_order=segment_data['position_order'],
                segment_metadata=segment_data.get('metadata', {})
            )
            db.add(segment)
        
        # Step 5: Generate metadata
        summary = await generate_summary(text, "section")
        key_terms = await extract_key_terms(text)
        reading_time = estimate_reading_time(text)
        
        metadata = ContentMetadata(
            content_id=uuid.UUID(content_id),
            key_terms=key_terms,
            estimated_duration_mins=reading_time,
            summary=summary
        )
        db.add(metadata)
        
        # Step 6: Mark as completed
        content.processing_status = ProcessingStatus.COMPLETED
        db.commit()
        
    except Exception as e:
        print(f"Processing failed: {e}")
        content.processing_status = ProcessingStatus.FAILED
        db.commit()

# ============= Routes =============

@app.post("/api/v1/content/upload", response_model=ContentUploadResponse)
async def upload_content(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = "",
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    
    user_id = uuid.uuid4()  # Mock - would come from auth
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain", "text/html"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Read file
    file_content = await file.read()
    file_size = len(file_content)
    
    # Validate size (max 50MB)
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    # Upload to S3
    storage_url = upload_to_s3(file_content, file.filename, file.content_type)
    
    # Create content record
    content = Content(
        user_id=user_id,
        title=title or file.filename,
        content_type=file.content_type,
        storage_url=storage_url,
        file_size_bytes=file_size,
        processing_status=ProcessingStatus.PENDING
    )
    db.add(content)
    db.commit()
    db.refresh(content)
    
    # Schedule background processing
    background_tasks.add_task(
        process_content_background,
        str(content.id),
        file_content,
        file.content_type,
        db
    )
    
    return ContentUploadResponse(
        content_id=str(content.id),
        title=content.title,
        storage_url=content.storage_url,
        processing_status=content.processing_status,
        created_at=content.created_at.isoformat()
    )

@app.get("/api/v1/content/{content_id}")
async def get_content(content_id: str, db: Session = Depends(get_db)):
    """Get content details"""
    
    content = db.query(Content).filter(Content.id == uuid.UUID(content_id)).first()
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    return {
        "id": str(content.id),
        "title": content.title,
        "content_type": content.content_type,
        "storage_url": content.storage_url,
        "file_size_bytes": content.file_size_bytes,
        "processing_status": content.processing_status,
        "created_at": content.created_at.isoformat()
    }

@app.get("/api/v1/content/{content_id}/segments")
async def get_content_segments(
    content_id: str,
    importance: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get content segments"""
    
    query = db.query(ContentSegment).filter(ContentSegment.content_id == uuid.UUID(content_id))
    
    if importance:
        query = query.filter(ContentSegment.importance_level == importance)
    
    segments = query.order_by(ContentSegment.position_order).all()
    
    return {
        "content_id": content_id,
        "total_segments": len(segments),
        "segments": [
            {
                "id": str(s.id),
                "segment_type": s.segment_type,
                "text_content": s.text_content,
                "importance_level": s.importance_level,
                "position_order": s.position_order,
        "metadata": s.segment_metadata
            }
            for s in segments
        ]
    }

@app.get("/api/v1/content/{content_id}/summary")
async def get_content_summary(
    content_id: str,
    level: str = "section",
    db: Session = Depends(get_db)
):
    """Get content summary at specified detail level"""
    
    metadata = db.query(ContentMetadata).filter(ContentMetadata.content_id == uuid.UUID(content_id)).first()
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found. Content may still be processing.")
    
    # If requesting different level than stored, regenerate
    if level != "section":
        content = db.query(Content).filter(Content.id == uuid.UUID(content_id)).first()
        segments = db.query(ContentSegment)\
            .filter(ContentSegment.content_id == uuid.UUID(content_id))\
            .order_by(ContentSegment.position_order)\
            .all()
        
        full_text = "\n\n".join([s.text_content for s in segments])
        summary = await generate_summary(full_text, level)
    else:
        summary = metadata.summary
    
    return ContentSummary(
        content_id=content_id,
        summary=summary,
        key_terms=metadata.key_terms,
        reading_level=metadata.reading_level,
        estimated_duration_mins=metadata.estimated_duration_mins
    )

@app.get("/api/v1/content/{content_id}/outline")
async def get_content_outline(content_id: str, db: Session = Depends(get_db)):
    """Get hierarchical outline of content"""
    
    segments = db.query(ContentSegment)\
        .filter(ContentSegment.content_id == uuid.UUID(content_id))\
        .filter(ContentSegment.segment_type == SegmentType.HEADING)\
        .order_by(ContentSegment.position_order)\
        .all()
    
    outline = [
        {
            "level": 1,  # Would determine from heading level in real implementation
            "text": s.text_content,
            "position": s.position_order
        }
        for s in segments
    ]
    
    return {
        "content_id": content_id,
        "outline": outline
    }

@app.post("/api/v1/content/{content_id}/tag-for-assignment")
async def tag_for_assignment(
    content_id: str,
    request: TagForAssignmentRequest,
    db: Session = Depends(get_db)
):
    """Re-classify segments based on assignment focus"""
    
    segments = db.query(ContentSegment)\
        .filter(ContentSegment.content_id == uuid.UUID(content_id))\
        .order_by(ContentSegment.position_order)\
        .all()
    
    if not segments:
        raise HTTPException(status_code=404, detail="No segments found")
    
    # Convert to dict format for classification
    segment_dicts = [
        {
            "position_order": s.position_order,
            "text_content": s.text_content,
            "segment_type": s.segment_type
        }
        for s in segments
    ]
    
    # Reclassify with assignment focus
    classified = await classify_importance_with_ai(segment_dicts, request.assignment_focus)
    
    # Update database
    for idx, segment in enumerate(segments):
        if idx < len(classified):
            segment.importance_level = classified[idx].get('importance_level')
            segment.segment_metadata = {
                **(segment.segment_metadata or {}),
                "assignment_focused": True,
                "assignment_focus": request.assignment_focus,
                "classification_reason": classified[idx].get('metadata', {}).get('classification_reason')
            }
    
    db.commit()
    
    return {
        "success": True,
        "content_id": content_id,
        "reclassified_segments": len(classified),
        "assignment_focus": request.assignment_focus
    }

@app.get("/api/v1/content")
async def list_content(
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List user's content"""
    
    query = db.query(Content)
    
    if user_id:
        query = query.filter(Content.user_id == uuid.UUID(user_id))
    
    total = query.count()
    content_list = query.order_by(Content.created_at.desc()).limit(limit).offset(offset).all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "content": [
            {
                "id": str(c.id),
                "title": c.title,
                "content_type": c.content_type,
                "processing_status": c.processing_status,
                "created_at": c.created_at.isoformat()
            }
            for c in content_list
        ]
    }

@app.delete("/api/v1/content/{content_id}")
async def delete_content(content_id: str, db: Session = Depends(get_db)):
    """Delete content and all associated data"""
    
    content = db.query(Content).filter(Content.id == uuid.UUID(content_id)).first()
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    # Delete from S3
    try:
        # Extract key from URL
        key = content.storage_url.split(f"{S3_BUCKET}.s3.amazonaws.com/")[1]
        s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
    except Exception as e:
        print(f"S3 deletion failed: {e}")
    
    # Delete segments
    db.query(ContentSegment).filter(ContentSegment.content_id == uuid.UUID(content_id)).delete()
    
    # Delete metadata
    db.query(ContentMetadata).filter(ContentMetadata.content_id == uuid.UUID(content_id)).delete()
    
    # Delete content record
    db.delete(content)
    db.commit()
    
    return {"success": True, "content_id": content_id}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "content-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)