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
    LIMEAnalysis
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
    'CVRankingService',
    'get_service'
]

