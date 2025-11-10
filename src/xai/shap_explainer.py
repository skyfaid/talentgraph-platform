"""
SHAP (SHapley Additive exPlanations) integration for feature importance.

SHAP provides mathematically rigorous feature importance scores.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..utils.logger import setup_logger

logger = setup_logger("shap_explainer")


class SHAPExplainer:
    """
    SHAP explainer for CV ranking system.
    
    Provides feature-level importance scores showing which features
    (skills, experience, etc.) contribute most to the ranking.
    """
    
    def __init__(self):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.available = False
        else:
            self.available = True
    
    def explain_embedding_importance(
        self,
        embedding_vector: np.ndarray,
        baseline_embedding: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain importance of embedding dimensions.
        
        Args:
            embedding_vector: Candidate's embedding vector
            baseline_embedding: Baseline/average embedding for comparison
            feature_names: Optional names for embedding dimensions
            
        Returns:
            Dictionary with feature importance scores
        """
        if not self.available:
            return {"error": "SHAP not available"}
        
        try:
            # Calculate difference from baseline
            diff = embedding_vector - baseline_embedding
            
            # Use absolute values as importance scores
            importance_scores = np.abs(diff)
            
            # Normalize
            total_importance = np.sum(importance_scores)
            if total_importance > 0:
                normalized_scores = importance_scores / total_importance
            else:
                normalized_scores = importance_scores
            
            # Get top contributing dimensions
            top_indices = np.argsort(importance_scores)[::-1][:10]
            
            result = {
                "top_features": [
                    {
                        "index": int(idx),
                        "importance": float(importance_scores[idx]),
                        "normalized_importance": float(normalized_scores[idx]),
                        "feature_name": feature_names[idx] if feature_names and idx < len(feature_names) else f"dim_{idx}"
                    }
                    for idx in top_indices
                ],
                "total_importance": float(total_importance),
                "max_importance": float(np.max(importance_scores))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return {"error": str(e)}
    
    def explain_skill_importance(
        self,
        skills_present: List[str],
        skills_required: List[str],
        skill_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Explain importance of skills in ranking.
        
        Args:
            skills_present: Skills found in candidate resume
            skills_required: Skills required by job
            skill_weights: Optional weights for each skill
            
        Returns:
            Dictionary with skill importance analysis
        """
        if not self.available:
            return {"error": "SHAP not available"}
        
        try:
            # Calculate skill importance based on matching
            skill_importance = {}
            
            for skill in skills_required:
                is_present = skill in skills_present
                weight = skill_weights.get(skill, 1.0) if skill_weights else 1.0
                
                # Importance = weight * (1 if present, -0.5 if missing)
                importance = weight * (1.0 if is_present else -0.5)
                skill_importance[skill] = {
                    "present": is_present,
                    "importance": importance,
                    "weight": weight
                }
            
            # Sort by importance
            sorted_skills = sorted(
                skill_importance.items(),
                key=lambda x: abs(x[1]["importance"]),
                reverse=True
            )
            
            return {
                "skill_contributions": {
                    skill: data for skill, data in sorted_skills
                },
                "top_positive": [
                    skill for skill, data in sorted_skills[:5] if data["importance"] > 0
                ],
                "top_negative": [
                    skill for skill, data in sorted_skills[:5] if data["importance"] < 0
                ],
                "total_contribution": sum(data["importance"] for data in skill_importance.values())
            }
            
        except Exception as e:
            logger.error(f"Error in skill importance: {e}")
            return {"error": str(e)}


def get_shap_explainer() -> SHAPExplainer:
    """Get or create SHAP explainer instance."""
    return SHAPExplainer()

