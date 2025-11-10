"""
Interview Session Manager.

Manages interview sessions:
- Create and track interview sessions
- Store questions and answers
- Calculate overall scores
- Manage session state
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid

from ..utils.logger import setup_logger
from ..utils.config import BASE_DIR

logger = setup_logger("session_manager")


class InterviewSessionManager:
    """
    Manages interview sessions.
    
    Stores sessions in JSON files (can be upgraded to database later).
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize session manager.
        
        Args:
            storage_dir: Directory to store session files. Default: BASE_DIR / "interview_sessions"
        """
        if storage_dir is None:
            storage_dir = BASE_DIR / "interview_sessions"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        logger.info(f"InterviewSessionManager initialized, storage: {self.storage_dir}")
    
    def create_session(
        self,
        candidate_id: str,
        candidate_name: str,
        candidate_email: str,
        job_description: str,
        candidate_resume: str,
        interview_type: str = "mixed"
    ) -> str:
        """
        Create a new interview session.
        
        Args:
            candidate_id: Candidate ID from CV ranking
            candidate_name: Candidate name
            candidate_email: Candidate email
            job_description: Job description
            candidate_resume: Candidate resume text
            interview_type: "technical", "behavioral", or "mixed"
        
        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        
        session = {
            "session_id": session_id,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "candidate_email": candidate_email,
            "job_description": job_description,
            "candidate_resume": candidate_resume,
            "interview_type": interview_type,
            "status": "created",  # created, in_progress, completed
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "questions": [],
            "answers": [],
            "evaluations": [],
            "overall_score": None,
            "technical_score": None,
            "behavioral_score": None
        }
        
        self._save_session(session)
        logger.info(f"Created interview session: {session_id} for candidate: {candidate_name}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        session_file = self.storage_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def add_question(self, session_id: str, question: Dict[str, Any]) -> bool:
        """Add a question to the session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        question["question_id"] = len(session["questions"])
        question["added_at"] = datetime.now().isoformat()
        session["questions"].append(question)
        session["updated_at"] = datetime.now().isoformat()
        session["status"] = "in_progress"
        
        self._save_session(session)
        return True
    
    def add_answer(
        self,
        session_id: str,
        question_id: int,
        answer: str,
        evaluation: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an answer and evaluation to the session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        answer_data = {
            "question_id": question_id,
            "answer": answer,
            "evaluation": evaluation,
            "answered_at": datetime.now().isoformat()
        }
        
        session["answers"].append(answer_data)
        
        # Update evaluation if provided
        if evaluation:
            if question_id < len(session["evaluations"]):
                session["evaluations"][question_id] = evaluation
            else:
                # Extend list if needed
                while len(session["evaluations"]) <= question_id:
                    session["evaluations"].append(None)
                session["evaluations"][question_id] = evaluation
        
        session["updated_at"] = datetime.now().isoformat()
        self._save_session(session)
        
        return True
    
    def calculate_overall_scores(self, session_id: str) -> Dict[str, float]:
        """
        Calculate overall interview scores.
        
        Returns:
            Dictionary with:
            - overall_score: Average of all question scores
            - technical_score: Average of technical question scores
            - behavioral_score: Average of behavioral question scores
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        scores = {
            "overall_score": 0.0,
            "technical_score": 0.0,
            "behavioral_score": 0.0
        }
        
        technical_scores = []
        behavioral_scores = []
        all_scores = []
        
        for i, evaluation in enumerate(session.get("evaluations", [])):
            if evaluation and "score" in evaluation:
                score = evaluation["score"]
                all_scores.append(score)
                
                # Check question type
                if i < len(session["questions"]):
                    question = session["questions"][i]
                    if question.get("type") == "technical":
                        technical_scores.append(score)
                    elif question.get("type") == "behavioral":
                        behavioral_scores.append(score)
        
        # Calculate averages
        if all_scores:
            scores["overall_score"] = sum(all_scores) / len(all_scores)
        
        if technical_scores:
            scores["technical_score"] = sum(technical_scores) / len(technical_scores)
        
        if behavioral_scores:
            scores["behavioral_score"] = sum(behavioral_scores) / len(behavioral_scores)
        
        # Update session
        session["overall_score"] = scores["overall_score"]
        session["technical_score"] = scores["technical_score"]
        session["behavioral_score"] = scores["behavioral_score"]
        session["updated_at"] = datetime.now().isoformat()
        
        self._save_session(session)
        
        return scores
    
    def complete_session(self, session_id: str) -> bool:
        """Mark session as completed."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["status"] = "completed"
        session["updated_at"] = datetime.now().isoformat()
        session["completed_at"] = datetime.now().isoformat()
        
        # Calculate final scores
        self.calculate_overall_scores(session_id)
        
        self._save_session(session)
        logger.info(f"Completed interview session: {session_id}")
        
        return True
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session (for reports)."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        scores = self.calculate_overall_scores(session_id)
        
        return {
            "session_id": session_id,
            "candidate_name": session["candidate_name"],
            "candidate_email": session["candidate_email"],
            "status": session["status"],
            "interview_type": session["interview_type"],
            "total_questions": len(session["questions"]),
            "total_answers": len(session["answers"]),
            "scores": scores,
            "created_at": session["created_at"],
            "completed_at": session.get("completed_at")
        }
    
    def _save_session(self, session: Dict[str, Any]):
        """Save session to file."""
        session_file = self.storage_dir / f"{session['session_id']}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session['session_id']}: {e}")
            raise


# Singleton instance
_session_manager = None


def get_session_manager(storage_dir=None) -> InterviewSessionManager:
    """
    Get or create session manager instance (singleton).
    
    Args:
        storage_dir: Optional storage directory
    
    Returns:
        InterviewSessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = InterviewSessionManager(storage_dir=storage_dir)
    return _session_manager

