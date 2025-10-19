# sdk/learning_companion/client.py
"""
Learning Companion Python SDK
Provides easy-to-use client for all API services
"""
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

@dataclass
class Config:
    """SDK Configuration"""
    base_url: str = "https://api.learningcompanion.ai"
    api_key: Optional[str] = None
    timeout: int = 30

class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, status_code: int, message: str, details: Any = None):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(f"API Error {status_code}: {message}")

class BaseClient:
    """Base client with common functionality"""
    
    def __init__(self, config: Config, access_token: Optional[str] = None):
        self.config = config
        self.access_token = access_token
        self.session = requests.Session()
        
        if access_token:
            self.session.headers.update({
                "Authorization": f"Bearer {access_token}"
            })
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None
    ) -> Any:
        """Make HTTP request"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data if data and not files else None,
                params=params,
                files=files,
                timeout=self.config.timeout
            )
            
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    raise APIError(
                        response.status_code,
                        error_data.get("detail", "Unknown error"),
                        error_data
                    )
                except json.JSONDecodeError:
                    raise APIError(response.status_code, response.text)
            
            return response.json() if response.text else None
            
        except requests.RequestException as e:
            raise APIError(500, f"Request failed: {str(e)}")

class AuthClient(BaseClient):
    """Authentication client"""
    
    def register(
        self,
        email: str,
        password: str,
        display_name: str,
        role: str = "student",
        grade_level: Optional[str] = None
    ) -> Dict:
        """Register a new user"""
        return self._request(
            "POST",
            "/api/v1/auth/register",
            data={
                "email": email,
                "password": password,
                "display_name": display_name,
                "role": role,
                "grade_level": grade_level
            }
        )
    
    def login(self, email: str, password: str) -> Dict:
        """Login and get tokens"""
        response = self._request(
            "POST",
            "/api/v1/auth/login",
            data={"email": email, "password": password}
        )
        
        # Update session with new token
        if "access_token" in response:
            self.access_token = response["access_token"]
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}"
            })
        
        return response
    
    def refresh_token(self, refresh_token: str) -> Dict:
        """Refresh access token"""
        return self._request(
            "POST",
            "/api/v1/auth/refresh",
            data={"refresh_token": refresh_token}
        )

class UserClient(BaseClient):
    """User management client"""
    
    def get_me(self) -> Dict:
        """Get current user profile"""
        return self._request("GET", "/api/v1/users/me")
    
    def update_profile(self, **kwargs) -> Dict:
        """Update user profile"""
        return self._request("PATCH", "/api/v1/users/me/profile", data=kwargs)
    
    def get_learning_profile(self) -> Dict:
        """Get learning profile"""
        return self._request("GET", "/api/v1/users/me/learning-profile")
    
    def update_learning_profile(self, **kwargs) -> Dict:
        """Update learning profile"""
        return self._request("PATCH", "/api/v1/users/me/learning-profile", data=kwargs)

class AIClient(BaseClient):
    """AI interaction client"""
    
    def ask_question(self, question: str, context_ids: List[str] = None) -> Dict:
        """Ask a question"""
        return self._request(
            "POST",
            "/api/v1/ai/question",
            data={
                "question": question,
                "context_ids": context_ids or []
            }
        )
    
    def explain_concept(self, topic: str, context: str = "", detail_level: str = "medium") -> Dict:
        """Get explanation of a concept"""
        return self._request(
            "POST",
            "/api/v1/ai/explain",
            data={
                "topic": topic,
                "context": context,
                "detail_level": detail_level
            }
        )
    
    def check_work(self, problem: str, solution: str, subject: str = "math") -> Dict:
        """Check student work for errors"""
        return self._request(
            "POST",
            "/api/v1/ai/check-work",
            data={
                "problem": problem,
                "solution_steps": solution,
                "subject": subject
            }
        )
    
    def generate_hint(self, problem: str, current_work: str, hint_level: str = "moderate") -> Dict:
        """Generate a hint"""
        return self._request(
            "POST",
            "/api/v1/ai/generate-hint",
            data={
                "problem": problem,
                "current_work": current_work,
                "hint_level": hint_level
            }
        )
    
    def summarize(self, content: str, assignment_focus: str = "", detail_level: str = "section") -> Dict:
        """Summarize content"""
        return self._request(
            "POST",
            "/api/v1/ai/summarize",
            data={
                "content": content,
                "assignment_focus": assignment_focus,
                "detail_level": detail_level
            }
        )
    
    def create_chat_session(self, session_type: str = "study_buddy", initial_context: Dict = None) -> Dict:
        """Create a chat session"""
        return self._request(
            "POST",
            "/api/v1/ai/chat/session",
            data={
                "session_type": session_type,
                "initial_context": initial_context
            }
        )
    
    def send_chat_message(self, session_id: str, message: str) -> Dict:
        """Send message in chat session"""
        return self._request(
            "POST",
            f"/api/v1/ai/chat/session/{session_id}/message",
            data={"message": message}
        )

class ContentClient(BaseClient):
    """Content management client"""
    
    def upload(self, file_path: str, title: str = "") -> Dict:
        """Upload a document"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            return self._request(
                "POST",
                "/api/v1/content/upload",
                params={"title": title},
                files=files
            )
    
    def get(self, content_id: str) -> Dict:
        """Get content details"""
        return self._request("GET", f"/api/v1/content/{content_id}")
    
    def get_segments(self, content_id: str, importance: str = None) -> Dict:
        """Get content segments"""
        params = {"importance": importance} if importance else None
        return self._request("GET", f"/api/v1/content/{content_id}/segments", params=params)
    
    def get_summary(self, content_id: str, level: str = "section") -> Dict:
        """Get content summary"""
        return self._request("GET", f"/api/v1/content/{content_id}/summary", params={"level": level})
    
    def tag_for_assignment(self, content_id: str, assignment_focus: str, rubric: Dict = None) -> Dict:
        """Tag content for an assignment"""
        return self._request(
            "POST",
            f"/api/v1/content/{content_id}/tag-for-assignment",
            data={
                "assignment_focus": assignment_focus,
                "rubric": rubric
            }
        )
    
    def list(self, limit: int = 50, offset: int = 0) -> Dict:
        """List user's content"""
        return self._request("GET", "/api/v1/content", params={"limit": limit, "offset": offset})
    
    def delete(self, content_id: str) -> Dict:
        """Delete content"""
        return self._request("DELETE", f"/api/v1/content/{content_id}")

class LabClient(BaseClient):
    """Lab/sandbox client"""
    
    def start_lab(self, lab_type: str, template_id: str = None) -> Dict:
        """Start a lab session"""
        return self._request(
            "POST",
            "/api/v1/labs/start",
            data={
                "lab_type": lab_type,
                "template_id": template_id
            }
        )
    
    def validate_circuit(self, session_id: str, circuit: Dict, validation_type: str) -> Dict:
        """Validate circuit"""
        return self._request(
            "POST",
            f"/api/v1/labs/{session_id}/validate",
            data={
                "circuit": circuit,
                "validation_type": validation_type
            }
        )
    
    def measure(self, session_id: str, circuit: Dict, measurement_type: str, **kwargs) -> Dict:
        """Take measurements"""
        return self._request(
            "POST",
            f"/api/v1/labs/{session_id}/measure",
            data={
                "circuit": circuit,
                "measurement_type": measurement_type,
                **kwargs
            }
        )
    
    def complete_lab(self, session_id: str, final_score: float, mastery_achieved: bool) -> Dict:
        """Complete lab session"""
        return self._request(
            "POST",
            f"/api/v1/labs/{session_id}/complete",
            data={
                "final_score": final_score,
                "mastery_achieved": mastery_achieved
            }
        )

class GamificationClient(BaseClient):
    """Gamification client"""
    
    def get_progress(self, user_id: str = "me") -> Dict:
        """Get user progress"""
        return self._request("GET", f"/api/v1/progress/{user_id}")
    
    def award_xp(self, user_id: str, activity_type: str, context: Dict = None) -> Dict:
        """Award XP for an activity"""
        return self._request(
            "POST",
            f"/api/v1/progress/{user_id}/award-xp",
            data={
                "activity_type": activity_type,
                "context": context or {}
            }
        )
    
    def get_achievements(self, user_id: str = "me") -> Dict:
        """Get user achievements"""
        return self._request("GET", f"/api/v1/progress/{user_id}/achievements")
    
    def get_roadmap(self, user_id: str = "me") -> Dict:
        """Get learning roadmap"""
        return self._request("GET", f"/api/v1/progress/{user_id}/roadmap")
    
    def start_roadmap_node(self, user_id: str, node_id: str) -> Dict:
        """Start a roadmap node"""
        return self._request("POST", f"/api/v1/progress/{user_id}/roadmap/{node_id}/start")
    
    def complete_roadmap_node(self, user_id: str, node_id: str) -> Dict:
        """Complete a roadmap node"""
        return self._request("POST", f"/api/v1/progress/{user_id}/roadmap/{node_id}/complete")
    
    def get_leaderboard(self, metric: str = "xp", scope: str = "global", limit: int = 10) -> Dict:
        """Get leaderboard"""
        return self._request(
            "GET",
            "/api/v1/leaderboard",
            params={
                "metric": metric,
                "scope": scope,
                "limit": limit
            }
        )

class AnalyticsClient(BaseClient):
    """Analytics client"""
    
    def get_student_report(self, user_id: str, days: int = 30) -> Dict:
        """Get student report"""
        return self._request(
            "GET",
            f"/api/v1/analytics/student/{user_id}/report",
            params={"days": days}
        )
    
    def get_class_overview(self, class_id: str, days: int = 7) -> Dict:
        """Get class overview"""
        return self._request(
            "GET",
            f"/api/v1/analytics/class/{class_id}/overview",
            params={"days": days}
        )
    
    def get_at_risk_students(self, class_id: str, days: int = 14) -> Dict:
        """Get at-risk students"""
        return self._request(
            "GET",
            f"/api/v1/analytics/class/{class_id}/at-risk",
            params={"days": days}
        )
    
    def get_common_struggles(self, class_id: str, limit: int = 10) -> Dict:
        """Get common struggle topics"""
        return self._request(
            "GET",
            f"/api/v1/analytics/class/{class_id}/common-struggles",
            params={"limit": limit}
        )

class LearningCompanionClient:
    """Main SDK client with all sub-clients"""
    
    def __init__(
        self,
        base_url: str = "https://api.learningcompanion.ai",
        api_key: Optional[str] = None,
        access_token: Optional[str] = None
    ):
        self.config = Config(base_url=base_url, api_key=api_key)
        
        # Initialize all sub-clients
        self.auth = AuthClient(self.config, access_token)
        self.users = UserClient(self.config, access_token)
        self.ai = AIClient(self.config, access_token)
        self.content = ContentClient(self.config, access_token)
        self.labs = LabClient(self.config, access_token)
        self.gamification = GamificationClient(self.config, access_token)
        self.analytics = AnalyticsClient(self.config, access_token)
    
    def set_access_token(self, token: str):
        """Set access token for all clients"""
        for client_name in ['auth', 'users', 'ai', 'content', 'labs', 'gamification', 'analytics']:
            client = getattr(self, client_name)
            client.access_token = token
            client.session.headers.update({"Authorization": f"Bearer {token}"})

# ============= Usage Examples =============

def example_usage():
    """Examples of using the SDK"""
    
    # Initialize client
    client = LearningCompanionClient(base_url="https://api.learningcompanion.ai")
    
    # Register and login
    try:
        # Register
        user = client.auth.register(
            email="student@example.com",
            password="securepassword",
            display_name="John Doe",
            role="student",
            grade_level="Grade 10"
        )
        print(f"Registered user: {user['id']}")
        
        # Login
        tokens = client.auth.login(
            email="student@example.com",
            password="securepassword"
        )
        print(f"Logged in, access token: {tokens['access_token'][:20]}...")
        
        # Set token for subsequent requests
        client.set_access_token(tokens['access_token'])
        
        # Get user profile
        profile = client.users.get_me()
        print(f"User profile: {profile['display_name']}")
        
        # Ask AI a question
        response = client.ai.ask_question("What is photosynthesis?")
        print(f"AI response: {response['response'][:100]}...")
        
        # Check math work
        feedback = client.ai.check_work(
            problem="Solve: 2x + 5 = 13",
            solution="2x = 8\nx = 4",
            subject="math"
        )
        print(f"Work feedback: {feedback}")
        
        # Upload content
        content = client.content.upload(
            file_path="./biology_chapter1.pdf",
            title="Biology Chapter 1"
        )
        print(f"Uploaded content: {content['content_id']}")
        
        # Get summary
        summary = client.content.get_summary(content['content_id'], level="overview")
        print(f"Summary: {summary['summary']}")
        
        # Start a lab
        lab = client.labs.start_lab(lab_type="circuit", template_id=None)
        print(f"Started lab: {lab['session_id']}")
        
        # Get progress
        progress = client.gamification.get_progress("me")
        print(f"Progress: Level {progress['level']}, {progress['total_xp']} XP")
        
        # Get student report
        report = client.analytics.get_student_report("me", days=30)
        print(f"Report: {report['engagement']}")
        
    except APIError as e:
        print(f"API Error: {e.message}")
        print(f"Details: {e.details}")

if __name__ == "__main__":
    example_usage()