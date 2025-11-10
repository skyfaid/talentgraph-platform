"""
Explainable AI (XAI) modules for CV ranking system.

Includes:
- RankingExplainer: Main explainer with rule-based + SHAP + LIME
- SHAPExplainer: Feature-level importance
- LIMEExplainer: Text-level importance
"""
from .explainer import RankingExplainer, explain_ranking

try:
    from .shap_explainer import SHAPExplainer, get_shap_explainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    SHAPExplainer = None
    get_shap_explainer = None

try:
    from .lime_explainer import LIMEExplainer, get_lime_explainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    LIMEExplainer = None
    get_lime_explainer = None

__all__ = [
    'RankingExplainer',
    'explain_ranking',
    'SHAPExplainer',
    'LIMEExplainer',
    'get_shap_explainer',
    'get_lime_explainer',
    'SHAP_AVAILABLE',
    'LIME_AVAILABLE'
]

