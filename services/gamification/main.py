# services/gamification/main.py
"""
Gamification Service - XP, Achievements, Levels, and Progress Tracking
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON, Boolean, DECIMAL, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
import os
from datetime import datetime, date, timedelta
from enum import Enum

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/learning_companion")

# Database Setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Gamification Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Database Models =============

class UserProgress(Base):
    __tablename__ = "user_progress"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    total_xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    current_streak_days = Column(Integer, default=0)
    longest_streak_days = Column(Integer, default=0)
    last_activity_date = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Achievement(Base):
    __tablename__ = "achievements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(String(1000))
    requirement = Column(JSON, nullable=False)
    reward_xp = Column(Integer, default=0)
    badge_image_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    category = Column(String(50))
    tier = Column(String(20))  # bronze, silver, gold, platinum
    created_at = Column(DateTime, default=datetime.utcnow)

class UserAchievement(Base):
    __tablename__ = "user_achievements"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    achievement_id = Column(UUID(as_uuid=True), primary_key=True)
    earned_at = Column(DateTime, default=datetime.utcnow)
    notified = Column(Boolean, default=False)

class RoadmapNode(Base):
    __tablename__ = "roadmap_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_name = Column(String(255), nullable=False)
    node_type = Column(String(50), nullable=False)
    description = Column(String(1000))
    prerequisites = Column(JSON, default=[])
    xp_to_unlock = Column(Integer, default=0)
    content_id = Column(UUID(as_uuid=True))
    estimated_duration_mins = Column(Integer)
    position = Column(JSON)  # {x, y} for visual map
    created_at = Column(DateTime, default=datetime.utcnow)

class UserRoadmapProgress(Base):
    __tablename__ = "user_roadmap_progress"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    node_id = Column(UUID(as_uuid=True), primary_key=True)
    status = Column(String(50), default="locked")  # locked, available, in_progress, completed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

class XPTransaction(Base):
    __tablename__ = "xp_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    reason = Column(String(255))
    related_entity_type = Column(String(50))
    related_entity_id = Column(UUID(as_uuid=True))
    timestamp = Column(DateTime, default=datetime.utcnow)

# ============= Pydantic Models =============

class ActivityType(str, Enum):
    READING_COMPLETE = "reading_complete"
    PRACTICE_PROBLEM = "practice_problem"
    LAB_COMPLETION = "lab_completion"
    QUIZ_PASSED = "quiz_passed"
    MISTAKE_CORRECTED = "mistake_corrected"
    DAILY_STREAK = "daily_streak"
    ASSIGNMENT_SUBMITTED = "assignment_submitted"
    HELP_PEER = "help_peer"

class AwardXPRequest(BaseModel):
    activity_type: ActivityType
    context: Dict[str, Any] = {}

class AchievementCreate(BaseModel):
    name: str
    description: str
    requirement: Dict[str, Any]
    reward_xp: int = 0
    category: str = "general"
    tier: str = "bronze"
    badge_image_url: Optional[str] = None

class ProgressResponse(BaseModel):
    user_id: str
    total_xp: int
    level: int
    xp_for_current_level: int
    xp_for_next_level: int
    xp_progress_percentage: float
    current_streak_days: int
    longest_streak_days: int

class AchievementResponse(BaseModel):
    id: str
    name: str
    description: str
    reward_xp: int
    category: str
    tier: str
    earned_at: Optional[str]
    is_earned: bool

class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    display_name: str
    metric_value: int
    level: int

# ============= XP System Configuration =============

BASE_XP = {
    ActivityType.READING_COMPLETE: 10,
    ActivityType.PRACTICE_PROBLEM: 15,
    ActivityType.LAB_COMPLETION: 25,
    ActivityType.QUIZ_PASSED: 20,
    ActivityType.MISTAKE_CORRECTED: 5,
    ActivityType.DAILY_STREAK: 10,
    ActivityType.ASSIGNMENT_SUBMITTED: 30,
    ActivityType.HELP_PEER: 15,
}

def calculate_xp_for_level(level: int) -> int:
    """Calculate total XP needed to reach a level"""
    # Formula: XP = 100 * level^1.5
    return int(100 * (level ** 1.5))

def get_level_from_xp(total_xp: int) -> int:
    """Calculate level from total XP"""
    level = 1
    while calculate_xp_for_level(level + 1) <= total_xp:
        level += 1
    return level

# ============= XP Calculator =============

class XPCalculator:
    """Calculate XP with multipliers based on context"""
    
    @staticmethod
    def calculate(activity_type: ActivityType, context: Dict[str, Any]) -> int:
        base = BASE_XP.get(activity_type, 10)
        
        # Difficulty multiplier
        difficulty = context.get("difficulty", 1.0)
        
        # First try bonus
        first_try = 1.2 if context.get("attempts", 999) == 1 else 1.0
        
        # Streak bonus (up to 1.5x)
        streak_days = context.get("streak_days", 0)
        streak_bonus = min(1.5, 1.0 + (streak_days * 0.05))
        
        # Speed bonus for quick completion
        expected_time = context.get("expected_time_mins", 30)
        actual_time = context.get("actual_time_mins", 30)
        speed_bonus = 1.0
        if actual_time < expected_time * 0.75:
            speed_bonus = 1.1
        
        # Accuracy bonus
        accuracy = context.get("accuracy", 0.0)
        accuracy_bonus = 1.0 + (accuracy * 0.2) if accuracy > 0.8 else 1.0
        
        total = int(base * difficulty * first_try * streak_bonus * speed_bonus * accuracy_bonus)
        
        return max(1, total)  # Minimum 1 XP

# ============= Achievement Checker =============

class AchievementChecker:
    """Check if user has earned new achievements"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_all_achievements(self, user_id: uuid.UUID) -> List[Achievement]:
        """Check all achievements for a user"""
        
        # Get user's current progress
        progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        if not progress:
            return []
        
        # Get earned achievements
        earned_ids = [
            ua.achievement_id 
            for ua in self.db.query(UserAchievement).filter(UserAchievement.user_id == user_id).all()
        ]
        
        # Get all active achievements not yet earned
        available_achievements = self.db.query(Achievement)\
            .filter(Achievement.is_active == True)\
            .filter(~Achievement.id.in_(earned_ids))\
            .all()
        
        newly_earned = []
        
        for achievement in available_achievements:
            if self._check_requirement(user_id, achievement.requirement):
                # Award achievement
                user_achievement = UserAchievement(
                    user_id=user_id,
                    achievement_id=achievement.id
                )
                self.db.add(user_achievement)
                
                # Award XP
                if achievement.reward_xp > 0:
                    progress.total_xp += achievement.reward_xp
                    
                    transaction = XPTransaction(
                        user_id=user_id,
                        amount=achievement.reward_xp,
                        reason=f"Achievement: {achievement.name}",
                        related_entity_type="achievement",
                        related_entity_id=achievement.id
                    )
                    self.db.add(transaction)
                
                newly_earned.append(achievement)
        
        if newly_earned:
            self.db.commit()
        
        return newly_earned
    
    def _check_requirement(self, user_id: uuid.UUID, requirement: Dict[str, Any]) -> bool:
        """Check if user meets achievement requirement"""
        
        req_type = requirement.get("type")
        
        if req_type == "xp_total":
            progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
            return progress and progress.total_xp >= requirement.get("value", 0)
        
        elif req_type == "streak_days":
            progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
            return progress and progress.current_streak_days >= requirement.get("value", 0)
        
        elif req_type == "level":
            progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
            return progress and progress.level >= requirement.get("value", 0)
        
        elif req_type == "lessons_completed":
            # Would query from study sessions
            count = requirement.get("value", 0)
            # Mock check for now
            return False
        
        elif req_type == "problems_solved":
            # Would query from study buddy interactions
            count = requirement.get("value", 0)
            return False
        
        return False

# ============= Dependencies =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= Routes =============

@app.get("/api/v1/progress/{user_id}", response_model=ProgressResponse)
async def get_user_progress(user_id: str, db: Session = Depends(get_db)):
    """Get user's progress and level"""
    
    progress = db.query(UserProgress).filter(UserProgress.user_id == uuid.UUID(user_id)).first()
    
    if not progress:
        # Create new progress record
        progress = UserProgress(user_id=uuid.UUID(user_id))
        db.add(progress)
        db.commit()
        db.refresh(progress)
    
    # Calculate XP for levels
    current_level_xp = calculate_xp_for_level(progress.level)
    next_level_xp = calculate_xp_for_level(progress.level + 1)
    xp_in_current_level = progress.total_xp - current_level_xp
    xp_needed_for_next = next_level_xp - current_level_xp
    
    progress_percentage = (xp_in_current_level / xp_needed_for_next) * 100 if xp_needed_for_next > 0 else 0
    
    return ProgressResponse(
        user_id=str(progress.user_id),
        total_xp=progress.total_xp,
        level=progress.level,
        xp_for_current_level=current_level_xp,
        xp_for_next_level=next_level_xp,
        xp_progress_percentage=round(progress_percentage, 1),
        current_streak_days=progress.current_streak_days,
        longest_streak_days=progress.longest_streak_days
    )

@app.post("/api/v1/progress/{user_id}/award-xp")
async def award_xp(user_id: str, request: AwardXPRequest, db: Session = Depends(get_db)):
    """Award XP to a user for an activity"""
    
    progress = db.query(UserProgress).filter(UserProgress.user_id == uuid.UUID(user_id)).first()
    
    if not progress:
        progress = UserProgress(user_id=uuid.UUID(user_id))
        db.add(progress)
    
    # Update streak if needed
    today = date.today()
    if progress.last_activity_date:
        last_date = progress.last_activity_date.date()
        if last_date == today:
            pass  # Same day, no streak update
        elif last_date == today - timedelta(days=1):
            # Consecutive day
            progress.current_streak_days += 1
            progress.longest_streak_days = max(progress.longest_streak_days, progress.current_streak_days)
        else:
            # Streak broken
            progress.current_streak_days = 1
    else:
        progress.current_streak_days = 1
    
    progress.last_activity_date = datetime.utcnow()
    
    # Add streak to context
    request.context["streak_days"] = progress.current_streak_days
    
    # Calculate XP
    xp_earned = XPCalculator.calculate(request.activity_type, request.context)
    
    # Award XP
    old_xp = progress.total_xp
    progress.total_xp += xp_earned
    
    # Update level
    new_level = get_level_from_xp(progress.total_xp)
    level_up = new_level > progress.level
    progress.level = new_level
    
    # Log transaction
    transaction = XPTransaction(
        user_id=uuid.UUID(user_id),
        amount=xp_earned,
        reason=request.activity_type.value,
        related_entity_type=request.context.get("entity_type"),
        related_entity_id=uuid.UUID(request.context["entity_id"]) if request.context.get("entity_id") else None
    )
    db.add(transaction)
    
    db.commit()
    
    # Check for new achievements
    checker = AchievementChecker(db)
    new_achievements = checker.check_all_achievements(uuid.UUID(user_id))
    
    return {
        "success": True,
        "xp_earned": xp_earned,
        "total_xp": progress.total_xp,
        "level": progress.level,
        "level_up": level_up,
        "new_achievements": [
            {"id": str(a.id), "name": a.name, "reward_xp": a.reward_xp}
            for a in new_achievements
        ]
    }

@app.get("/api/v1/progress/{user_id}/history")
async def get_xp_history(
    user_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get user's XP transaction history"""
    
    transactions = db.query(XPTransaction)\
        .filter(XPTransaction.user_id == uuid.UUID(user_id))\
        .order_by(XPTransaction.timestamp.desc())\
        .limit(limit)\
        .all()
    
    return {
        "user_id": user_id,
        "transactions": [
            {
                "id": str(t.id),
                "amount": t.amount,
                "reason": t.reason,
                "timestamp": t.timestamp.isoformat()
            }
            for t in transactions
        ]
    }

@app.post("/api/v1/achievements", response_model=AchievementResponse)
async def create_achievement(request: AchievementCreate, db: Session = Depends(get_db)):
    """Create a new achievement (admin only)"""
    
    achievement = Achievement(
        name=request.name,
        description=request.description,
        requirement=request.requirement,
        reward_xp=request.reward_xp,
        category=request.category,
        tier=request.tier,
        badge_image_url=request.badge_image_url
    )
    
    db.add(achievement)
    db.commit()
    db.refresh(achievement)
    
    return AchievementResponse(
        id=str(achievement.id),
        name=achievement.name,
        description=achievement.description,
        reward_xp=achievement.reward_xp,
        category=achievement.category,
        tier=achievement.tier,
        earned_at=None,
        is_earned=False
    )

@app.get("/api/v1/achievements")
async def list_achievements(
    category: Optional[str] = None,
    tier: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all achievements"""
    
    query = db.query(Achievement).filter(Achievement.is_active == True)
    
    if category:
        query = query.filter(Achievement.category == category)
    
    if tier:
        query = query.filter(Achievement.tier == tier)
    
    achievements = query.all()
    
    return {
        "achievements": [
            {
                "id": str(a.id),
                "name": a.name,
                "description": a.description,
                "reward_xp": a.reward_xp,
                "category": a.category,
                "tier": a.tier,
                "requirement": a.requirement
            }
            for a in achievements
        ]
    }

@app.get("/api/v1/progress/{user_id}/achievements")
async def get_user_achievements(user_id: str, db: Session = Depends(get_db)):
    """Get user's achievements"""
    
    # Get all achievements
    all_achievements = db.query(Achievement).filter(Achievement.is_active == True).all()
    
    # Get user's earned achievements
    earned = db.query(UserAchievement)\
        .filter(UserAchievement.user_id == uuid.UUID(user_id))\
        .all()
    
    earned_map = {str(e.achievement_id): e.earned_at for e in earned}
    
    achievements = []
    for achievement in all_achievements:
        achievement_id = str(achievement.id)
        achievements.append(AchievementResponse(
            id=achievement_id,
            name=achievement.name,
            description=achievement.description,
            reward_xp=achievement.reward_xp,
            category=achievement.category,
            tier=achievement.tier,
            earned_at=earned_map[achievement_id].isoformat() if achievement_id in earned_map else None,
            is_earned=achievement_id in earned_map
        ))
    
    # Sort: earned first, then by tier/XP
    achievements.sort(key=lambda x: (not x.is_earned, -x.reward_xp))
    
    return {
        "user_id": user_id,
        "total_earned": len(earned_map),
        "total_available": len(all_achievements),
        "achievements": achievements
    }

@app.get("/api/v1/leaderboard")
async def get_leaderboard(
    metric: str = "xp",
    scope: str = "global",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get leaderboard"""
    
    # For now, implement XP leaderboard
    if metric == "xp":
        top_users = db.query(UserProgress)\
            .order_by(UserProgress.total_xp.desc())\
            .limit(limit)\
            .all()
        
        leaderboard = []
        for rank, user_progress in enumerate(top_users, start=1):
            leaderboard.append(LeaderboardEntry(
                rank=rank,
                user_id=str(user_progress.user_id),
                display_name=f"User {str(user_progress.user_id)[:8]}",  # Would fetch from user service
                metric_value=user_progress.total_xp,
                level=user_progress.level
            ))
        
        return {
            "metric": metric,
            "scope": scope,
            "leaderboard": leaderboard
        }
    
    elif metric == "streak":
        top_users = db.query(UserProgress)\
            .order_by(UserProgress.current_streak_days.desc())\
            .limit(limit)\
            .all()
        
        leaderboard = []
        for rank, user_progress in enumerate(top_users, start=1):
            leaderboard.append(LeaderboardEntry(
                rank=rank,
                user_id=str(user_progress.user_id),
                display_name=f"User {str(user_progress.user_id)[:8]}",
                metric_value=user_progress.current_streak_days,
                level=user_progress.level
            ))
        
        return {
            "metric": metric,
            "scope": scope,
            "leaderboard": leaderboard
        }

@app.post("/api/v1/roadmap/nodes")
async def create_roadmap_node(
    node_name: str,
    node_type: str,
    description: str = "",
    prerequisites: List[str] = [],
    xp_to_unlock: int = 0,
    db: Session = Depends(get_db)
):
    """Create a roadmap node"""
    
    node = RoadmapNode(
        node_name=node_name,
        node_type=node_type,
        description=description,
        prerequisites=prerequisites,
        xp_to_unlock=xp_to_unlock
    )
    
    db.add(node)
    db.commit()
    db.refresh(node)
    
    return {
        "id": str(node.id),
        "node_name": node.node_name,
        "node_type": node.type,
        "prerequisites": node.prerequisites,
        "xp_to_unlock": node.xp_to_unlock
    }

@app.get("/api/v1/progress/{user_id}/roadmap")
async def get_user_roadmap(user_id: str, db: Session = Depends(get_db)):
    """Get user's roadmap progress"""
    
    # Get all nodes
    nodes = db.query(RoadmapNode).all()
    
    # Get user progress
    user_progress = db.query(UserProgress).filter(UserProgress.user_id == uuid.UUID(user_id)).first()
    
    # Get node statuses
    progress_records = db.query(UserRoadmapProgress)\
        .filter(UserRoadmapProgress.user_id == uuid.UUID(user_id))\
        .all()
    
    progress_map = {str(p.node_id): p for p in progress_records}
    
    roadmap = []
    for node in nodes:
        node_id = str(node.id)
        progress_record = progress_map.get(node_id)
        
        # Determine status
        if progress_record:
            status = progress_record.status
        elif user_progress and user_progress.total_xp >= node.xp_to_unlock:
            status = "available"
        else:
            status = "locked"
        
        roadmap.append({
            "id": node_id,
            "name": node.node_name,
            "type": node.node_type,
            "description": node.description,
            "prerequisites": node.prerequisites,
            "xp_to_unlock": node.xp_to_unlock,
            "status": status,
            "completed_at": progress_record.completed_at.isoformat() if progress_record and progress_record.completed_at else None
        })
    
    return {
        "user_id": user_id,
        "roadmap": roadmap
    }

@app.post("/api/v1/progress/{user_id}/roadmap/{node_id}/start")
async def start_roadmap_node(user_id: str, node_id: str, db: Session = Depends(get_db)):
    """Start a roadmap node"""
    
    progress = db.query(UserRoadmapProgress)\
        .filter(UserRoadmapProgress.user_id == uuid.UUID(user_id))\
        .filter(UserRoadmapProgress.node_id == uuid.UUID(node_id))\
        .first()
    
    if not progress:
        progress = UserRoadmapProgress(
            user_id=uuid.UUID(user_id),
            node_id=uuid.UUID(node_id),
            status="in_progress",
            started_at=datetime.utcnow()
        )
        db.add(progress)
    else:
        progress.status = "in_progress"
        progress.started_at = datetime.utcnow()
    
    db.commit()
    
    return {"success": True, "node_id": node_id, "status": "in_progress"}

@app.post("/api/v1/progress/{user_id}/roadmap/{node_id}/complete")
async def complete_roadmap_node(user_id: str, node_id: str, db: Session = Depends(get_db)):
    """Complete a roadmap node"""
    
    progress = db.query(UserRoadmapProgress)\
        .filter(UserRoadmapProgress.user_id == uuid.UUID(user_id))\
        .filter(UserRoadmapProgress.node_id == uuid.UUID(node_id))\
        .first()
    
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    
    progress.status = "completed"
    progress.completed_at = datetime.utcnow()
    db.commit()
    
    return {"success": True, "node_id": node_id, "status": "completed"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gamification"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)