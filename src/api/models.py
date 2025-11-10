"""
FastAPI request and response models.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class RankRequest(BaseModel):
    """Request model for ranking CVs."""
    job_description: str = Field(..., description="Job description to match against")
    top_k: int = Field(5, ge=1, le=50, description="Number of top candidates to return")
    include_explanations: bool = Field(False, description="Whether to include XAI explanations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "Senior Data Engineer with Python, SQL, AWS; 5+ years; leadership a plus.",
                "top_k": 5,
                "include_explanations": False
            }
        }


class CandidateResult(BaseModel):
    """Model for a single ranked candidate."""
    final_score: float = Field(..., description="Final hybrid score (0-10)")
    llm_score: float = Field(..., description="LLM evaluation score (0-10)")
    semantic_similarity: float = Field(..., description="Semantic similarity score (0-1)")
    evaluation: str = Field(..., description="Full LLM evaluation text")
    meta: Dict[str, Any] = Field(..., description="Candidate metadata (id, name, email, category, source)")
    preview: str = Field(..., description="First 240 characters of resume")
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_score": 8.5,
                "llm_score": 9.0,
                "semantic_similarity": 0.85,
                "evaluation": "Score: 9/10. The candidate matches Python and AWS skills...",
                "meta": {
                    "id": "resume1_123",
                    "name": "John Smith",
                    "email": "john.smith123@gmail.com",
                    "category": "INFORMATION-TECHNOLOGY",
                    "source": "Resume.csv"
                },
                "preview": "software developer professional summary..."
            }
        }


class RankResponse(BaseModel):
    """Response model for ranking endpoint."""
    job_description: str = Field(..., description="The job description that was ranked against")
    top_k: int = Field(..., description="Number of candidates requested")
    candidates: List[Any] = Field(..., description="Ranked list of candidates (with explanations if requested)")
    total_candidates_evaluated: int = Field(..., description="Total number of candidates evaluated")
    
    class Config:
        json_encoders = {
            # Ensure Pydantic models are properly serialized
            object: lambda v: v.dict() if hasattr(v, 'dict') else (v.model_dump() if hasattr(v, 'model_dump') else v)
        }
        json_schema_extra = {
            "example": {
                "job_description": "Senior Data Engineer...",
                "top_k": 5,
                "candidates": [],
                "total_candidates_evaluated": 12
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    vectorstore_ready: bool = Field(..., description="Whether vectorstore is ready")
    llm_ready: bool = Field(..., description="Whether LLM is ready")
    total_resumes: Optional[int] = Field(None, description="Total resumes in vectorstore")


# ========== INTERVIEW MODELS ==========

class InterviewStartRequest(BaseModel):
    """Request model for starting an interview."""
    candidate_id: str = Field(..., description="Candidate ID from CV ranking")
    candidate_name: str = Field(..., description="Candidate name")
    candidate_email: str = Field(..., description="Candidate email")
    job_description: str = Field(..., description="Job description")
    candidate_resume: str = Field(..., description="Candidate resume text")
    interview_type: str = Field("mixed", description="Interview type: technical, behavioral, or mixed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidate_id": "resume1_123",
                "candidate_name": "John Smith",
                "candidate_email": "john@example.com",
                "job_description": "Senior Data Engineer...",
                "candidate_resume": "John Smith is a data engineer...",
                "interview_type": "mixed"
            }
        }


class InterviewStartResponse(BaseModel):
    """Response model for starting an interview."""
    session_id: str = Field(..., description="Interview session ID")
    status: str = Field(..., description="Session status")
    message: str = Field(..., description="Status message")


class QuestionResponse(BaseModel):
    """Response model for a question."""
    question_id: int = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    question_type: str = Field(..., description="Question type: technical, behavioral, or followup")
    expected_points: Optional[str] = Field(None, description="Expected answer points (for technical)")
    star_required: Optional[str] = Field(None, description="STAR format requirements (for behavioral)")
    skills_tested: Optional[List[str]] = Field(None, description="Skills being tested")
    skills_assessed: Optional[List[str]] = Field(None, description="Skills being assessed (for behavioral)")


class SubmitAnswerRequest(BaseModel):
    """Request model for submitting an answer."""
    session_id: str = Field(..., description="Interview session ID")
    question_id: int = Field(..., description="Question ID")
    answer: str = Field(..., description="Candidate's answer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc-123-def",
                "question_id": 0,
                "answer": "I have 5 years of experience with Python..."
            }
        }


class AnswerEvaluationResponse(BaseModel):
    """Response model for answer evaluation."""
    question_id: int = Field(..., description="Question ID")
    score: float = Field(..., description="Answer score (0-10)")
    strengths: List[str] = Field(..., description="Answer strengths")
    weaknesses: List[str] = Field(..., description="Answer weaknesses")
    key_points: List[str] = Field(..., description="Key points extracted")
    recommendation: str = Field(..., description="Recommendation: strong/weak/needs_followup")
    feedback: str = Field(..., description="Detailed feedback")
    star_completeness: Optional[str] = Field(None, description="STAR completeness (for behavioral)")


class InterviewStatusResponse(BaseModel):
    """Response model for interview status."""
    session_id: str = Field(..., description="Session ID")
    status: str = Field(..., description="Session status: created, in_progress, completed")
    candidate_name: str = Field(..., description="Candidate name")
    total_questions: int = Field(..., description="Total questions")
    answered_questions: int = Field(..., description="Number of answered questions")
    current_question_id: Optional[int] = Field(None, description="Current question ID (if in progress)")
    overall_score: Optional[float] = Field(None, description="Overall score (if completed)")
    technical_score: Optional[float] = Field(None, description="Technical score average")
    behavioral_score: Optional[float] = Field(None, description="Behavioral score average")


class InterviewReportResponse(BaseModel):
    """Response model for interview report."""
    session_id: str = Field(..., description="Session ID")
    candidate_name: str = Field(..., description="Candidate name")
    candidate_email: str = Field(..., description="Candidate email")
    job_description: str = Field(..., description="Job description")
    interview_type: str = Field(..., description="Interview type")
    status: str = Field(..., description="Session status")
    scores: Dict[str, float] = Field(..., description="Overall scores")
    questions: List[Dict[str, Any]] = Field(..., description="All questions")
    answers: List[Dict[str, Any]] = Field(..., description="All answers with evaluations")
    created_at: str = Field(..., description="Session creation time")
    completed_at: Optional[str] = Field(None, description="Session completion time")


class UploadResponse(BaseModel):
    """Response model for PDF upload."""
    message: str = Field(..., description="Upload status message")
    files_processed: int = Field(..., description="Number of files successfully processed")
    files_failed: int = Field(..., description="Number of files that failed")
    total_resumes: int = Field(..., description="Total resumes now in system")


class SkillAnalysis(BaseModel):
    """Skill matching analysis."""
    matched_skills: List[str] = Field(..., description="Skills that match between resume and job")
    missing_skills: List[str] = Field(..., description="Skills required by job but not in resume")
    extra_skills: List[str] = Field(..., description="Skills in resume but not required by job")
    match_rate: float = Field(..., description="Percentage of job skills matched")
    total_job_skills: int = Field(..., description="Total skills required by job")
    total_resume_skills: int = Field(..., description="Total skills in resume")


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown."""
    final_score: float = Field(..., description="Final hybrid score")
    semantic_contribution: float = Field(..., description="Contribution from semantic similarity")
    llm_contribution: float = Field(..., description="Contribution from LLM evaluation")
    semantic_score: float = Field(..., description="Normalized semantic score (0-10)")
    llm_score: float = Field(..., description="LLM score (0-10)")
    formula: str = Field(..., description="Formula used for calculation")


class SHAPAnalysis(BaseModel):
    """SHAP feature importance analysis."""
    available: bool = Field(..., description="Whether SHAP analysis is available")
    skill_importance: Optional[Dict[str, Any]] = Field(None, description="SHAP skill importance scores")
    error: Optional[str] = Field(None, description="Error message if SHAP failed")


class LIMEAnalysis(BaseModel):
    """LIME text-level importance analysis."""
    available: bool = Field(..., description="Whether LIME analysis is available")
    text_importance: Optional[Dict[str, Any]] = Field(None, description="LIME word/phrase importance")
    error: Optional[str] = Field(None, description="Error message if LIME failed")


class Explanation(BaseModel):
    """Complete explanation for a candidate ranking."""
    candidate_id: str = Field(..., description="Candidate ID")
    overall_score: float = Field(..., description="Overall ranking score")
    score_breakdown: ScoreBreakdown = Field(..., description="Detailed score breakdown")
    skill_analysis: SkillAnalysis = Field(..., description="Skill matching analysis")
    llm_evaluation: str = Field(..., description="Full LLM evaluation text")
    strengths: List[str] = Field(..., description="Candidate strengths")
    weaknesses: List[str] = Field(..., description="Candidate weaknesses/gaps")
    experience_info: Dict[str, Any] = Field(..., description="Experience level information")
    leadership_info: Dict[str, Any] = Field(..., description="Leadership information")
    primary_factor: str = Field(..., description="Primary factor influencing ranking")
    secondary_factors: List[str] = Field(..., description="Secondary ranking factors")
    shap_analysis: Optional[SHAPAnalysis] = Field(None, description="SHAP feature importance (if available)")
    lime_analysis: Optional[LIMEAnalysis] = Field(None, description="LIME text-level importance (if available)")


class ExplainRequest(BaseModel):
    """Request model for explanation endpoint."""
    candidate_id: str = Field(..., description="ID of candidate to explain")
    job_description: str = Field(..., description="Job description used for ranking")


class CandidateResultWithExplanation(CandidateResult):
    """Candidate result with XAI explanation."""
    explanation: Optional[Explanation] = Field(None, description="XAI explanation (if requested)")

