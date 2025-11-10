"""
FastAPI API modules.
"""
from .models import (
    RankRequest,
    RankResponse,
    CandidateResult,
    CandidateResultWithExplanation,
    HealthResponse,
    UploadResponse,
    Explanation,
    ExplainRequest,
    SkillAnalysis,
    ScoreBreakdown,
    SHAPAnalysis,
    LIMEAnalysis,
    # Interview models
    InterviewStartRequest,
    InterviewStartResponse,
    QuestionResponse,
    SubmitAnswerRequest,
    AnswerEvaluationResponse,
    InterviewStatusResponse,
    InterviewReportResponse
)
from .service import CVRankingService, get_service

__all__ = [
    'RankRequest',
    'RankResponse',
    'CandidateResult',
    'CandidateResultWithExplanation',
    'HealthResponse',
    'UploadResponse',
    'Explanation',
    'ExplainRequest',
    'SkillAnalysis',
    'ScoreBreakdown',
    'SHAPAnalysis',
    'LIMEAnalysis',
    # Interview models
    'InterviewStartRequest',
    'InterviewStartResponse',
    'QuestionResponse',
    'SubmitAnswerRequest',
    'AnswerEvaluationResponse',
    'InterviewStatusResponse',
    'InterviewReportResponse',
    'CVRankingService',
    'get_service'
]

