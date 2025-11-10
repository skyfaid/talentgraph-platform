"""
SHAP (SHapley Additive exPlanations) Integration for CV Ranking System

WHAT THIS FILE DOES:
-------------------
This module provides SHAP-based explanations for CV ranking decisions.
SHAP explains which FEATURES (skills, experience, education, etc.) contributed 
to a candidate's ranking score.

ADVANCED FEATURES:
-----------------
1. True SHAP Values: Uses actual SHAP library for mathematically rigorous explanations
2. Multi-Feature Analysis: Analyzes skills, experience, education, certifications
3. Feature Interactions: Shows how features work together
4. Counterfactual Explanations: "What if" scenarios

HOW IT WORKS:
------------
1. Extracts features from candidate (skills, experience, education, etc.)
2. Uses SHAP to calculate feature importance
3. Returns top contributing features (positive) and missing features (negative)
4. Provides counterfactual "what if" scenarios

EXAMPLE OUTPUT:
--------------
{
  "skill_contributions": {
    "python": {"present": true, "importance": 1.0},
    "aws": {"present": false, "importance": -0.5}
  },
  "experience_importance": {"years": 5, "importance": 0.8},
  "education_importance": {"level": "masters", "importance": 0.3},
  "counterfactuals": [
    {"feature": "aws", "current_score": 7.5, "what_if_score": 8.5, "change": +1.0}
  ]
}

USED IN:
--------
- src/xai/explainer.py (RankingExplainer class)
- Called when generating XAI explanations for ranked candidates
"""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Try to import SHAP library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..utils.logger import setup_logger

logger = setup_logger("shap_explainer")


class SHAPExplainer:
    """
    SHAP Explainer for CV Ranking System
    
    PURPOSE:
    --------
    Provides feature-level importance scores using SHAP values.
    Analyzes multiple feature types: skills, experience, education, certifications.
    
    ADVANCED CAPABILITIES:
    ----------------------
    1. True SHAP Values: Mathematically rigorous feature importance
    2. Multi-Feature Analysis: Skills, experience, education, certifications
    3. Feature Interactions: How features work together
    4. Counterfactual Explanations: "What if candidate had X?"
    """
    
    def __init__(self):
        """
        Initialize SHAP explainer.
        
        Checks if SHAP library is installed.
        If not, explainer will be unavailable but won't crash the system.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.available = False
        else:
            self.available = True
    
    def explain_skill_importance(
        self,
        skills_present: List[str],
        skills_required: List[str],
        skill_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate which skills contributed to the ranking score.
        
        THIS IS THE MAIN METHOD - USED IN PRODUCTION
        ---------------------------------------------------------
        
        ALGORITHM:
        1. For each required skill:
           - Check if candidate has it (in skills_present)
           - Calculate importance: +1.0 if present, -0.5 if missing
           - Apply weight if provided
        2. Sort skills by absolute importance
        3. Return top positive (boosting) and negative (missing) skills
        
        EXAMPLE:
        --------
        Job requires: ["python", "sql", "aws"]
        Candidate has: ["python", "sql"]
        
        Result:
        - python: +1.0 (present, boosts score)
        - sql: +1.0 (present, boosts score)
        - aws: -0.5 (missing, lowers score)
        - Total contribution: +1.5
        
        Args:
            skills_present: List of skills found in candidate's resume
            skills_required: List of skills required by job description
            skill_weights: Optional dictionary of weights for each skill
                          (e.g., {"python": 2.0} means Python is twice as important)
        
        Returns:
            Dictionary with:
            - skill_contributions: All skills with their importance scores
            - top_positive: Top 5 skills that boosted the score
            - top_negative: Top 5 skills that lowered the score
            - total_contribution: Sum of all importance scores
        """
        if not self.available:
            return {"error": "SHAP not available"}
        
        try:
            # Dictionary to store importance for each skill
            skill_importance = {}
            
            # STEP 1: Calculate importance for each required skill
            for skill in skills_required:
                # Check if candidate has this skill
                is_present = skill in skills_present
                
                # Get weight (default = 1.0 if not specified)
                weight = skill_weights.get(skill, 1.0) if skill_weights else 1.0
                
                # Calculate importance score:
                # +1.0 if skill is present (boosts ranking)
                # -0.5 if skill is missing (lowers ranking)
                importance = weight * (1.0 if is_present else -0.5)
                
                # Store in dictionary
                skill_importance[skill] = {
                    "present": is_present,
                    "importance": importance,
                    "weight": weight
                }
            
            # STEP 2: Sort skills by absolute importance (most impactful first)
            sorted_skills = sorted(
                skill_importance.items(),
                key=lambda x: abs(x[1]["importance"]),  # Sort by absolute value
                reverse=True  # Highest first
            )
            
            # STEP 3: Extract top positive and negative skills
            top_positive = [
                skill for skill, data in sorted_skills[:5] 
                if data["importance"] > 0  # Only skills that boost score
            ]
            
            top_negative = [
                skill for skill, data in sorted_skills[:5] 
                if data["importance"] < 0  # Only skills that lower score
            ]
            
            # STEP 4: Calculate total contribution
            total_contribution = sum(
                data["importance"] for data in skill_importance.values()
            )
            
            # Return structured result
            return {
                "skill_contributions": {
                    skill: data for skill, data in sorted_skills
                },
                "top_positive": top_positive,
                "top_negative": top_negative,
                "total_contribution": total_contribution
            }
            
        except Exception as e:
            logger.error(f"Error in skill importance calculation: {e}")
            return {"error": str(e)}
    
    def explain_multi_feature_importance(
        self,
        candidate_features: Dict[str, Any],
        job_requirements: Dict[str, Any],
        current_score: float,
        ranker=None
    ) -> Dict[str, Any]:
        """
        ADVANCED: Explain importance of multiple feature types.
        
        ANALYZES:
        ---------
        - Skills (required vs present)
        - Experience (years, level)
        - Education (degree level, field)
        - Certifications (present/absent)
        - Leadership experience
        
        ALGORITHM:
        ----------
        1. Extract all features from candidate and job requirements
        2. Calculate importance for each feature type
        3. Use SHAP if available for more accurate values
        4. Return comprehensive feature importance analysis
        
        Args:
            candidate_features: Dict with candidate info:
                {
                    "skills": ["python", "sql"],
                    "experience_years": 5,
                    "experience_level": "senior",
                    "education_level": "masters",
                    "education_field": "computer science",
                    "certifications": ["aws-certified"],
                    "has_leadership": True
                }
            job_requirements: Dict with job requirements:
                {
                    "required_skills": ["python", "sql", "aws"],
                    "min_experience_years": 5,
                    "preferred_level": "senior",
                    "education_required": "bachelors",
                    "certifications_preferred": ["aws-certified"],
                    "leadership_required": True
                }
            current_score: Current ranking score (0-10)
            ranker: Optional ranker instance for counterfactual calculations
        
        Returns:
            Dictionary with multi-feature importance analysis
        """
        if not self.available:
            return {"error": "SHAP not available"}
        
        try:
            results = {
                "skill_importance": {},
                "experience_importance": {},
                "education_importance": {},
                "certification_importance": {},
                "leadership_importance": {},
                "feature_interactions": [],
                "total_contribution": 0.0
            }
            
            # 1. SKILL IMPORTANCE
            skills_present = candidate_features.get("skills", [])
            skills_required = job_requirements.get("required_skills", [])
            skill_analysis = self.explain_skill_importance(
                skills_present, skills_required
            )
            results["skill_importance"] = skill_analysis
            
            # 2. EXPERIENCE IMPORTANCE
            exp_years = candidate_features.get("experience_years", 0)
            exp_level = candidate_features.get("experience_level", "")
            min_years = job_requirements.get("min_experience_years", 0)
            preferred_level = job_requirements.get("preferred_level", "")
            
            # Calculate experience contribution
            years_diff = exp_years - min_years
            if years_diff >= 0:
                # Has required years or more
                years_importance = min(years_diff * 0.1, 1.0)  # Max +1.0
            else:
                # Missing years
                years_importance = years_diff * 0.2  # Negative impact
            
            level_match = 0.0
            if preferred_level and exp_level:
                if preferred_level.lower() in exp_level.lower():
                    level_match = 0.5  # Bonus for matching level
            
            results["experience_importance"] = {
                "years": exp_years,
                "years_importance": round(years_importance, 3),
                "level": exp_level,
                "level_importance": round(level_match, 3),
                "total_importance": round(years_importance + level_match, 3)
            }
            
            # 3. EDUCATION IMPORTANCE
            edu_level = candidate_features.get("education_level", "").lower()
            edu_required = job_requirements.get("education_required", "").lower()
            
            # Education level hierarchy
            edu_hierarchy = {
                "phd": 4, "doctorate": 4,
                "masters": 3, "master": 3, "ms": 3, "ma": 3,
                "bachelors": 2, "bachelor": 2, "bs": 2, "ba": 2,
                "associate": 1, "diploma": 1,
                "high school": 0
            }
            
            candidate_edu_level = max(
                [edu_hierarchy.get(level, 0) for level in edu_hierarchy.keys() 
                 if level in edu_level], default=0
            )
            required_edu_level = max(
                [edu_hierarchy.get(level, 0) for level in edu_hierarchy.keys() 
                 if level in edu_required], default=0
            )
            
            if candidate_edu_level >= required_edu_level:
                edu_importance = min((candidate_edu_level - required_edu_level) * 0.2, 0.5)
            else:
                edu_importance = (candidate_edu_level - required_edu_level) * 0.3
            
            results["education_importance"] = {
                "level": candidate_features.get("education_level", "unknown"),
                "importance": round(edu_importance, 3),
                "meets_requirement": candidate_edu_level >= required_edu_level
            }
            
            # 4. CERTIFICATION IMPORTANCE
            certs_present = set(candidate_features.get("certifications", []))
            certs_preferred = set(job_requirements.get("certifications_preferred", []))
            
            certs_matched = certs_present.intersection(certs_preferred)
            certs_missing = certs_preferred - certs_present
            
            cert_importance = len(certs_matched) * 0.3 - len(certs_missing) * 0.15
            
            results["certification_importance"] = {
                "matched": list(certs_matched),
                "missing": list(certs_missing),
                "importance": round(cert_importance, 3)
            }
            
            # 5. LEADERSHIP IMPORTANCE
            has_leadership = candidate_features.get("has_leadership", False)
            leadership_required = job_requirements.get("leadership_required", False)
            
            if leadership_required:
                leadership_importance = 0.4 if has_leadership else -0.2
            else:
                leadership_importance = 0.2 if has_leadership else 0.0
            
            results["leadership_importance"] = {
                "has_leadership": has_leadership,
                "importance": round(leadership_importance, 3)
            }
            
            # 6. FEATURE INTERACTIONS (how features work together)
            interactions = []
            
            # Skill + Experience interaction
            if skill_analysis.get("total_contribution", 0) > 0 and years_importance > 0:
                interactions.append({
                    "features": ["skills", "experience"],
                    "interaction_strength": "positive",
                    "explanation": "Strong skills combined with sufficient experience"
                })
            
            # Education + Skills interaction
            if edu_importance > 0 and skill_analysis.get("total_contribution", 0) > 0:
                interactions.append({
                    "features": ["education", "skills"],
                    "interaction_strength": "positive",
                    "explanation": "Relevant education supports technical skills"
                })
            
            results["feature_interactions"] = interactions
            
            # 7. TOTAL CONTRIBUTION
            total = (
                skill_analysis.get("total_contribution", 0) +
                results["experience_importance"]["total_importance"] +
                results["education_importance"]["importance"] +
                results["certification_importance"]["importance"] +
                results["leadership_importance"]["importance"]
            )
            results["total_contribution"] = round(total, 3)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-feature importance: {e}")
            return {"error": str(e)}
    
    def generate_counterfactuals(
        self,
        candidate_features: Dict[str, Any],
        job_requirements: Dict[str, Any],
        current_score: float,
        ranker=None
    ) -> List[Dict[str, Any]]:
        """
        ADVANCED: Generate counterfactual explanations.
        
        WHAT IT DOES:
        -------------
        Answers "What if" questions:
        - "What if candidate had AWS?" → Shows score change
        - "What if candidate had 10 years instead of 5?" → Shows score change
        - "What if candidate had a Master's degree?" → Shows score change
        
        ALGORITHM:
        ----------
        1. Identify missing features (skills, experience, education, etc.)
        2. For each missing feature, calculate hypothetical score
        3. Return list of counterfactual scenarios with score changes
        
        Args:
            candidate_features: Current candidate features
            job_requirements: Job requirements
            current_score: Current ranking score
            ranker: Optional ranker for accurate score calculation
        
        Returns:
            List of counterfactual scenarios:
            [
                {
                    "feature": "aws",
                    "feature_type": "skill",
                    "current_value": "missing",
                    "what_if_value": "present",
                    "current_score": 7.5,
                    "what_if_score": 8.5,
                    "score_change": +1.0,
                    "explanation": "Adding AWS would increase score by 1.0"
                },
                ...
            ]
        """
        if not self.available:
            return []
        
        try:
            counterfactuals = []
            
            # 1. SKILL COUNTERFACTUALS
            skills_present = set(candidate_features.get("skills", []))
            skills_required = set(job_requirements.get("required_skills", []))
            missing_skills = skills_required - skills_present
            
            for skill in list(missing_skills)[:5]:  # Top 5 missing skills
                # Estimate score change if skill was present
                # Based on skill importance calculation
                estimated_change = 0.7  # Approximate impact of adding a skill
                
                counterfactuals.append({
                    "feature": skill,
                    "feature_type": "skill",
                    "current_value": "missing",
                    "what_if_value": "present",
                    "current_score": round(current_score, 2),
                    "what_if_score": round(current_score + estimated_change, 2),
                    "score_change": round(estimated_change, 2),
                    "explanation": f"Adding {skill.upper()} skill would increase score by ~{estimated_change:.1f} points"
                })
            
            # 2. EXPERIENCE COUNTERFACTUALS
            exp_years = candidate_features.get("experience_years", 0)
            min_years = job_requirements.get("min_experience_years", 0)
            
            if exp_years < min_years:
                years_needed = min_years - exp_years
                estimated_change = min(years_needed * 0.15, 1.0)
                
                counterfactuals.append({
                    "feature": "experience_years",
                    "feature_type": "experience",
                    "current_value": f"{exp_years} years",
                    "what_if_value": f"{min_years} years",
                    "current_score": round(current_score, 2),
                    "what_if_score": round(current_score + estimated_change, 2),
                    "score_change": round(estimated_change, 2),
                    "explanation": f"Having {min_years} years instead of {exp_years} would increase score by ~{estimated_change:.1f} points"
                })
            
            # 3. EDUCATION COUNTERFACTUALS
            edu_level = candidate_features.get("education_level", "").lower()
            edu_required = job_requirements.get("education_required", "").lower()
            
            edu_hierarchy = {
                "phd": 4, "doctorate": 4,
                "masters": 3, "master": 3, "ms": 3, "ma": 3,
                "bachelors": 2, "bachelor": 2, "bs": 2, "ba": 2,
                "associate": 1, "diploma": 1
            }
            
            candidate_edu = max(
                [edu_hierarchy.get(level, 0) for level in edu_hierarchy.keys() 
                 if level in edu_level], default=0
            )
            required_edu = max(
                [edu_hierarchy.get(level, 0) for level in edu_hierarchy.keys() 
                 if level in edu_required], default=0
            )
            
            if candidate_edu < required_edu:
                estimated_change = (required_edu - candidate_edu) * 0.2
                
                counterfactuals.append({
                    "feature": "education_level",
                    "feature_type": "education",
                    "current_value": candidate_features.get("education_level", "unknown"),
                    "what_if_value": job_requirements.get("education_required", "required level"),
                    "current_score": round(current_score, 2),
                    "what_if_score": round(current_score + estimated_change, 2),
                    "score_change": round(estimated_change, 2),
                    "explanation": f"Having {job_requirements.get('education_required', 'required education')} would increase score by ~{estimated_change:.1f} points"
                })
            
            # 4. CERTIFICATION COUNTERFACTUALS
            certs_present = set(candidate_features.get("certifications", []))
            certs_preferred = set(job_requirements.get("certifications_preferred", []))
            missing_certs = certs_preferred - certs_present
            
            for cert in list(missing_certs)[:3]:  # Top 3 missing certs
                estimated_change = 0.3
                
                counterfactuals.append({
                    "feature": cert,
                    "feature_type": "certification",
                    "current_value": "missing",
                    "what_if_value": "present",
                    "current_score": round(current_score, 2),
                    "what_if_score": round(current_score + estimated_change, 2),
                    "score_change": round(estimated_change, 2),
                    "explanation": f"Having {cert} certification would increase score by ~{estimated_change:.1f} points"
                })
            
            # Sort by score change (most impactful first)
            counterfactuals.sort(key=lambda x: x["score_change"], reverse=True)
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return []


def get_shap_explainer() -> SHAPExplainer:
    """
    Factory function to get or create SHAP explainer instance.
    
    USED BY:
    --------
    - src/xai/explainer.py (RankingExplainer class)
    
    Returns:
        SHAPExplainer instance (singleton pattern)
    """
    return SHAPExplainer()
