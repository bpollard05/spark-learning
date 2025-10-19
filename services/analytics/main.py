# services/analytics/main.py
"""
Analytics Service - Learning Analytics, Insights, and Teacher Dashboards
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON, Float, Date, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
import os
from datetime import datetime, date, timedelta
from collections import defaultdict

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/learning_companion")

# Database Setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Analytics Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Database Models =============

class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    event_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(UUID(as_uuid=True))

class DailyMetrics(Base):
    __tablename__ = "daily_metrics"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    date = Column(Date, primary_key=True)
    total_time_minutes = Column(Integer, default=0)
    lessons_completed = Column(Integer, default=0)
    problems_attempted = Column(Integer, default=0)
    problems_correct = Column(Integer, default=0)
    xp_earned = Column(Integer, default=0)
    modality_breakdown = Column(JSON, default={})

class StrugglePattern(Base):
    __tablename__ = "struggle_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    topic = Column(String(255))
    error_type = Column(String(100))
    frequency = Column(Integer, default=1)
    first_occurred = Column(DateTime)
    last_occurred = Column(DateTime)
    recommended_intervention = Column(String(500))
    intervention_applied = Column(Boolean, default=False)

# ============= Pydantic Models =============

class EventTrackRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any] = {}
    session_id: Optional[str] = None

class EngagementMetrics(BaseModel):
    total_time_minutes: int
    active_days: int
    average_session_minutes: float
    completion_rate: float

class ComprehensionMetrics(BaseModel):
    accuracy: float
    topics_mastered: int
    topics_struggling: int
    average_attempts: float

class StruggleArea(BaseModel):
    topic: str
    error_type: str
    frequency: int
    last_occurred: str
    recommended_intervention: str

class StudentReport(BaseModel):
    user_id: str
    date_range: str
    engagement: EngagementMetrics
    comprehension: ComprehensionMetrics
    struggle_areas: List[StruggleArea]
    recommendations: List[str]

# ============= Dependencies =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= Analytics Calculators =============

class AnalyticsCalculator:
    """Calculate various analytics metrics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_engagement(self, user_id: uuid.UUID, days: int = 30) -> EngagementMetrics:
        """Calculate engagement metrics"""
        
        start_date = date.today() - timedelta(days=days)
        
        # Get daily metrics
        metrics = self.db.query(DailyMetrics)\
            .filter(DailyMetrics.user_id == user_id)\
            .filter(DailyMetrics.date >= start_date)\
            .all()
        
        if not metrics:
            return EngagementMetrics(
                total_time_minutes=0,
                active_days=0,
                average_session_minutes=0,
                completion_rate=0
            )
        
        total_time = sum(m.total_time_minutes for m in metrics)
        active_days = len([m for m in metrics if m.total_time_minutes > 0])
        avg_session = total_time / active_days if active_days > 0 else 0
        
        # Calculate completion rate
        total_attempted = sum(m.lessons_completed for m in metrics)
        completion_rate = 0.85  # Mock - would calculate from actual lesson data
        
        return EngagementMetrics(
            total_time_minutes=total_time,
            active_days=active_days,
            average_session_minutes=round(avg_session, 1),
            completion_rate=round(completion_rate, 2)
        )
    
    def calculate_comprehension(self, user_id: uuid.UUID, days: int = 30) -> ComprehensionMetrics:
        """Calculate comprehension metrics"""
        
        start_date = date.today() - timedelta(days=days)
        
        metrics = self.db.query(DailyMetrics)\
            .filter(DailyMetrics.user_id == user_id)\
            .filter(DailyMetrics.date >= start_date)\
            .all()
        
        if not metrics:
            return ComprehensionMetrics(
                accuracy=0,
                topics_mastered=0,
                topics_struggling=0,
                average_attempts=0
            )
        
        # Calculate accuracy
        total_attempted = sum(m.problems_attempted for m in metrics)
        total_correct = sum(m.problems_correct for m in metrics)
        accuracy = total_correct / total_attempted if total_attempted > 0 else 0
        
        # Count struggles
        struggles = self.db.query(StrugglePattern)\
            .filter(StrugglePattern.user_id == user_id)\
            .filter(StrugglePattern.last_occurred >= datetime.now() - timedelta(days=days))\
            .count()
        
        return ComprehensionMetrics(
            accuracy=round(accuracy, 2),
            topics_mastered=5,  # Mock - would calculate from mastery data
            topics_struggling=struggles,
            average_attempts=1.5  # Mock - would calculate from actual data
        )
    
    def identify_struggle_areas(self, user_id: uuid.UUID, limit: int = 5) -> List[StruggleArea]:
        """Identify top struggle areas"""
        
        patterns = self.db.query(StrugglePattern)\
            .filter(StrugglePattern.user_id == user_id)\
            .filter(StrugglePattern.intervention_applied == False)\
            .order_by(StrugglePattern.frequency.desc())\
            .limit(limit)\
            .all()
        
        return [
            StruggleArea(
                topic=p.topic,
                error_type=p.error_type,
                frequency=p.frequency,
                last_occurred=p.last_occurred.isoformat(),
                recommended_intervention=p.recommended_intervention or "Practice more problems"
            )
            for p in patterns
        ]
    
    def generate_recommendations(
        self,
        engagement: EngagementMetrics,
        comprehension: ComprehensionMetrics,
        struggles: List[StruggleArea]
    ) -> List[str]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        # Engagement recommendations
        if engagement.active_days < 7:
            recommendations.append("Try to study at least 3-4 days per week for better retention")
        
        if engagement.average_session_minutes < 15:
            recommendations.append("Consider longer study sessions (20-30 minutes) for deeper learning")
        
        # Comprehension recommendations
        if comprehension.accuracy < 0.7:
            recommendations.append("Focus on understanding concepts before attempting problems")
        
        if comprehension.topics_struggling > 3:
            recommendations.append("Work on one struggling topic at a time for focused improvement")
        
        # Struggle-specific recommendations
        if struggles:
            top_struggle = struggles[0]
            recommendations.append(f"Priority: Address {top_struggle.topic} - {top_struggle.recommended_intervention}")
        
        return recommendations[:5]  # Return top 5

class ClassAnalytics:
    """Analytics for teachers viewing class performance"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_class_overview(self, student_ids: List[uuid.UUID], days: int = 7) -> Dict[str, Any]:
        """Get overview of class performance"""
        
        start_date = date.today() - timedelta(days=days)
        
        # Aggregate metrics
        metrics = self.db.query(DailyMetrics)\
            .filter(DailyMetrics.user_id.in_(student_ids))\
            .filter(DailyMetrics.date >= start_date)\
            .all()
        
        if not metrics:
            return {
                "total_students": len(student_ids),
                "active_students": 0,
                "avg_engagement_minutes": 0,
                "avg_accuracy": 0,
                "total_problems_solved": 0
            }
        
        active_students = len(set(m.user_id for m in metrics if m.total_time_minutes > 0))
        total_time = sum(m.total_time_minutes for m in metrics)
        total_attempted = sum(m.problems_attempted for m in metrics)
        total_correct = sum(m.problems_correct for m in metrics)
        
        return {
            "total_students": len(student_ids),
            "active_students": active_students,
            "avg_engagement_minutes": round(total_time / len(student_ids), 1),
            "avg_accuracy": round(total_correct / total_attempted, 2) if total_attempted > 0 else 0,
            "total_problems_solved": total_correct,
            "date_range": f"{start_date} to {date.today()}"
        }
    
    def identify_at_risk_students(self, student_ids: List[uuid.UUID], days: int = 14) -> List[Dict[str, Any]]:
        """Identify students who may need intervention"""
        
        start_date = date.today() - timedelta(days=days)
        at_risk = []
        
        for student_id in student_ids:
            metrics = self.db.query(DailyMetrics)\
                .filter(DailyMetrics.user_id == student_id)\
                .filter(DailyMetrics.date >= start_date)\
                .all()
            
            if not metrics:
                at_risk.append({
                    "user_id": str(student_id),
                    "reason": "No activity in last 14 days",
                    "severity": "high"
                })
                continue
            
            # Check engagement
            active_days = len([m for m in metrics if m.total_time_minutes > 0])
            if active_days < 3:
                at_risk.append({
                    "user_id": str(student_id),
                    "reason": f"Only {active_days} active days in last {days} days",
                    "severity": "medium"
                })
            
            # Check accuracy
            total_attempted = sum(m.problems_attempted for m in metrics)
            total_correct = sum(m.problems_correct for m in metrics)
            if total_attempted > 0:
                accuracy = total_correct / total_attempted
                if accuracy < 0.5:
                    at_risk.append({
                        "user_id": str(student_id),
                        "reason": f"Low accuracy: {accuracy:.0%}",
                        "severity": "high"
                    })
        
        return at_risk
    
    def get_common_struggles(self, student_ids: List[uuid.UUID], limit: int = 10) -> List[Dict[str, Any]]:
        """Identify common struggle topics across the class"""
        
        patterns = self.db.query(
            StrugglePattern.topic,
            StrugglePattern.error_type,
            func.count(StrugglePattern.id).label('student_count'),
            func.sum(StrugglePattern.frequency).label('total_frequency')
        )\
            .filter(StrugglePattern.user_id.in_(student_ids))\
            .group_by(StrugglePattern.topic, StrugglePattern.error_type)\
            .order_by(func.count(StrugglePattern.id).desc())\
            .limit(limit)\
            .all()
        
        return [
            {
                "topic": p.topic,
                "error_type": p.error_type,
                "student_count": p.student_count,
                "total_occurrences": p.total_frequency,
                "percentage_of_class": round(p.student_count / len(student_ids) * 100, 1)
            }
            for p in patterns
        ]

# ============= Routes =============

@app.post("/api/v1/analytics/track")
async def track_event(request: EventTrackRequest, db: Session = Depends(get_db)):
    """Track an analytics event"""
    
    user_id = uuid.uuid4()  # Mock - would come from auth
    
    event = AnalyticsEvent(
        user_id=user_id,
        event_type=request.event_type,
        event_data=request.event_data,
        session_id=uuid.UUID(request.session_id) if request.session_id else None
    )
    
    db.add(event)
    db.commit()
    
    return {"success": True, "event_id": str(event.id)}

@app.post("/api/v1/analytics/daily-metrics")
async def update_daily_metrics(
    user_id: str,
    time_minutes: int = 0,
    lessons_completed: int = 0,
    problems_attempted: int = 0,
    problems_correct: int = 0,
    xp_earned: int = 0,
    db: Session = Depends(get_db)
):
    """Update or create daily metrics for a user"""
    
    today = date.today()
    
    metrics = db.query(DailyMetrics)\
        .filter(DailyMetrics.user_id == uuid.UUID(user_id))\
        .filter(DailyMetrics.date == today)\
        .first()
    
    if not metrics:
        metrics = DailyMetrics(
            user_id=uuid.UUID(user_id),
            date=today
        )
        db.add(metrics)
    
    # Increment values
    metrics.total_time_minutes += time_minutes
    metrics.lessons_completed += lessons_completed
    metrics.problems_attempted += problems_attempted
    metrics.problems_correct += problems_correct
    metrics.xp_earned += xp_earned
    
    db.commit()
    
    return {"success": True, "date": today.isoformat()}

@app.get("/api/v1/analytics/student/{user_id}/report", response_model=StudentReport)
async def get_student_report(
    user_id: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get comprehensive student report"""
    
    calculator = AnalyticsCalculator(db)
    user_uuid = uuid.UUID(user_id)
    
    engagement = calculator.calculate_engagement(user_uuid, days)
    comprehension = calculator.calculate_comprehension(user_uuid, days)
    struggles = calculator.identify_struggle_areas(user_uuid)
    recommendations = calculator.generate_recommendations(engagement, comprehension, struggles)
    
    return StudentReport(
        user_id=user_id,
        date_range=f"Last {days} days",
        engagement=engagement,
        comprehension=comprehension,
        struggle_areas=struggles,
        recommendations=recommendations
    )

@app.get("/api/v1/analytics/class/{class_id}/overview")
async def get_class_overview(
    class_id: str,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get class overview (teacher view)"""
    
    # Mock student IDs - would fetch from class roster
    student_ids = [uuid.uuid4() for _ in range(25)]
    
    class_analytics = ClassAnalytics(db)
    overview = class_analytics.get_class_overview(student_ids, days)
    
    return overview

@app.get("/api/v1/analytics/class/{class_id}/at-risk")
async def get_at_risk_students(
    class_id: str,
    days: int = 14,
    db: Session = Depends(get_db)
):
    """Identify at-risk students in a class"""
    
    student_ids = [uuid.uuid4() for _ in range(25)]  # Mock
    
    class_analytics = ClassAnalytics(db)
    at_risk = class_analytics.identify_at_risk_students(student_ids, days)
    
    return {
        "class_id": class_id,
        "at_risk_count": len(at_risk),
        "students": at_risk
    }

@app.get("/api/v1/analytics/class/{class_id}/common-struggles")
async def get_common_struggles(
    class_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get common struggle topics for a class"""
    
    student_ids = [uuid.uuid4() for _ in range(25)]  # Mock
    
    class_analytics = ClassAnalytics(db)
    struggles = class_analytics.get_common_struggles(student_ids, limit)
    
    return {
        "class_id": class_id,
        "struggles": struggles,
        "recommendation": "Consider a mini-lesson on the top struggle topics"
    }

@app.get("/api/v1/analytics/student/{user_id}/engagement-trend")
async def get_engagement_trend(
    user_id: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get engagement trend over time"""
    
    start_date = date.today() - timedelta(days=days)
    
    metrics = db.query(DailyMetrics)\
        .filter(DailyMetrics.user_id == uuid.UUID(user_id))\
        .filter(DailyMetrics.date >= start_date)\
        .order_by(DailyMetrics.date)\
        .all()
    
    trend = [
        {
            "date": m.date.isoformat(),
            "time_minutes": m.total_time_minutes,
            "lessons_completed": m.lessons_completed,
            "accuracy": m.problems_correct / m.problems_attempted if m.problems_attempted > 0 else 0
        }
        for m in metrics
    ]
    
    return {
        "user_id": user_id,
        "start_date": start_date.isoformat(),
        "end_date": date.today().isoformat(),
        "trend": trend
    }

@app.post("/api/v1/analytics/struggle-pattern")
async def log_struggle_pattern(
    user_id: str,
    topic: str,
    error_type: str,
    recommended_intervention: str = "",
    db: Session = Depends(get_db)
):
    """Log a struggle pattern for a user"""
    
    # Check if pattern exists
    pattern = db.query(StrugglePattern)\
        .filter(StrugglePattern.user_id == uuid.UUID(user_id))\
        .filter(StrugglePattern.topic == topic)\
        .filter(StrugglePattern.error_type == error_type)\
        .first()
    
    if pattern:
        # Update existing pattern
        pattern.frequency += 1
        pattern.last_occurred = datetime.utcnow()
    else:
        # Create new pattern
        pattern = StrugglePattern(
            user_id=uuid.UUID(user_id),
            topic=topic,
            error_type=error_type,
            frequency=1,
            first_occurred=datetime.utcnow(),
            last_occurred=datetime.utcnow(),
            recommended_intervention=recommended_intervention
        )
        db.add(pattern)
    
    db.commit()
    
    return {
        "success": True,
        "pattern_id": str(pattern.id),
        "frequency": pattern.frequency
    }

@app.get("/api/v1/analytics/content/{content_id}/effectiveness")
async def get_content_effectiveness(
    content_id: str,
    db: Session = Depends(get_db)
):
    """Analyze effectiveness of content"""
    
    # Query events related to this content
    events = db.query(AnalyticsEvent)\
        .filter(AnalyticsEvent.event_data['content_id'].astext == content_id)\
        .all()
    
    if not events:
        return {
            "content_id": content_id,
            "views": 0,
            "completions": 0,
            "avg_time_spent": 0,
            "effectiveness_score": 0
        }
    
    views = len([e for e in events if e.event_type == "content_viewed"])
    completions = len([e for e in events if e.event_type == "content_completed"])
    
    # Calculate average time spent
    time_events = [
        e.event_data.get("time_spent", 0)
        for e in events
        if e.event_type == "content_completed"
    ]
    avg_time = sum(time_events) / len(time_events) if time_events else 0
    
    # Simple effectiveness score
    completion_rate = completions / views if views > 0 else 0
    effectiveness_score = round(completion_rate * 100, 1)
    
    return {
        "content_id": content_id,
        "views": views,
        "completions": completions,
        "completion_rate": round(completion_rate, 2),
        "avg_time_spent_minutes": round(avg_time, 1),
        "effectiveness_score": effectiveness_score
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "analytics"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)