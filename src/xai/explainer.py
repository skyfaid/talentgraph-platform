"""
Explainable AI (XAI) module for CV ranking system.

Provides explanations for why candidates were ranked the way they were.
Uses hybrid approach: rule-based + SHAP + LIME for comprehensive explanations.
"""
import re
import numpy as np
from typing import Dict, List, Any, Optional
from ..utils.logger import setup_logger

# Import SHAP and LIME explainers
try:
    from .shap_explainer import SHAPExplainer, get_shap_explainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    get_shap_explainer = None

try:
    from .lime_explainer import LIMEExplainer, get_lime_explainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    get_lime_explainer = None

logger = setup_logger("xai_explainer")


class RankingExplainer:
    """
    Generates comprehensive explanations for CV ranking decisions.
    
    Uses hybrid approach:
    - Rule-based: Fast skill matching and score breakdowns
    - SHAP: Feature-level importance (when available)
    - LIME: Word/phrase-level importance (when available)
    
    Provides:
    - Skill matching analysis
    - Score breakdown explanations
    - Feature importance (SHAP)
    - Text-level importance (LIME)
    - Comparison insights
    """
    
    def __init__(self, use_shap: bool = True, use_lime: bool = True):
        """
        Initialize the explainer.
        
        Args:
            use_shap: Whether to use SHAP for feature importance
            use_lime: Whether to use LIME for text-level explanations
        """
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.use_lime = use_lime and LIME_AVAILABLE
        
        if self.use_shap:
            self.shap_explainer = get_shap_explainer() if get_shap_explainer else None
        else:
            self.shap_explainer = None
        
        if self.use_lime:
            self.lime_explainer = get_lime_explainer() if get_lime_explainer else None
        else:
            self.lime_explainer = None
        
        logger.info(
            f"RankingExplainer initialized: SHAP={self.use_shap}, LIME={self.use_lime}"
        )
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills mentioned in text using improved patterns.
        
        Args:
            text: Resume or job description text
            
        Returns:
            List of potential skills
        """
        # Expanded tech skills patterns - more comprehensive
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|sql|r|scala|go|rust|c\+\+|c#|php|ruby|swift|kotlin|perl|bash|shell|powershell)\b',
            # Cloud platforms
            r'\b(aws|amazon web services|azure|gcp|google cloud|oracle cloud|ibm cloud|bluemix)\b',
            # Containers & orchestration
            r'\b(docker|kubernetes|k8s|container|openshift|rancher)\b',
            # Web frameworks
            r'\b(react|vue|angular|node\.js|django|flask|fastapi|spring|express|nest\.js|next\.js|nuxt\.js)\b',
            # Data & ML frameworks
            r'\b(tensorflow|pytorch|pandas|numpy|scikit-learn|sklearn|spark|pyspark|hadoop|kafka|airflow|prefect|dbt|snowflake|databricks)\b',
            # Job titles & roles
            r'\b(data engineer|software engineer|data scientist|ml engineer|devops|backend|frontend|full stack|cloud architect|data analyst|business intelligence|bi engineer)\b',
            # Technologies & tools
            r'\b(machine learning|deep learning|nlp|natural language processing|computer vision|data pipeline|etl|data warehouse|data lake|big data|apache|git|jenkins|terraform|ansible|kubernetes|docker)\b',
            # Databases
            r'\b(mysql|postgresql|postgres|mongodb|cassandra|redis|elasticsearch|oracle|sql server|dynamodb|redshift|bigquery|snowflake)\b',
            # BI tools
            r'\b(tableau|power bi|looker|qlik|metabase|superset|grafana|kibana)\b'
        ]
        
        skills = set()
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            # Handle both single matches and tuple matches
            for match in matches:
                if isinstance(match, tuple):
                    skills.update([m for m in match if m])
                else:
                    skills.add(match)
        
        return sorted(list(skills))
    
    def extract_skills_from_llm_evaluation(self, llm_evaluation: str) -> Dict[str, List[str]]:
        """
        Extract skills from LLM evaluation text - more reliable than regex.
        Properly separates matched vs missing skills.
        
        Args:
            llm_evaluation: LLM evaluation text
            
        Returns:
            Dictionary with matched_skills, missing_skills extracted from LLM text
        """
        matched_skills = []
        missing_skills = []
        
        # Look for "Matched skills:" section - be more specific
        matched_patterns = [
            r'\*\*Matched skills?:\*\*\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Missing|Reasons)[^\n]+)*)',
            r'Matched skills?[:\-]\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Missing|Reasons)[^\n]+)*)',
            r'(?:has|possesses|experience with|mentioned)\s+([^\.\n]+(?:python|sql|aws|java|javascript|typescript|docker|kubernetes|react|vue|angular|tensorflow|pytorch|spark|hadoop|airflow)[^\.\n]*)',
        ]
        
        for pattern in matched_patterns:
            matched_section = re.search(pattern, llm_evaluation, re.IGNORECASE | re.MULTILINE)
            if matched_section:
                matched_text = matched_section.group(1)
                # Extract skills from the matched section
                extracted = self.extract_skills_from_text(matched_text)
                for skill in extracted:
                    if skill not in matched_skills:
                        matched_skills.append(skill)
                break  # Use first match
        
        # Look for "Missing skills:" section - be more specific
        missing_patterns = [
            r'\*\*Missing skills?:\*\*\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Reasons|Overall)[^\n]+)*)',
            r'Missing skills?[:\-]\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Reasons|Overall)[^\n]+)*)',
            r'(?:lacks?|missing|doesn\'t have|not mentioned|not explicitly|absent|no explicit)\s+([^\.\n]+(?:aws|airflow|spark|hadoop|python|sql|docker|kubernetes)[^\.\n]*)',
        ]
        
        for pattern in missing_patterns:
            missing_section = re.search(pattern, llm_evaluation, re.IGNORECASE | re.MULTILINE)
            if missing_section:
                missing_text = missing_section.group(1)
                # Extract skills from the missing section
                extracted = self.extract_skills_from_text(missing_text)
                for skill in extracted:
                    if skill not in missing_skills and skill not in matched_skills:  # Don't add if already matched
                        missing_skills.append(skill)
                break  # Use first match
        
        # Also check for skills mentioned with positive context (but not in missing)
        all_skills_in_eval = self.extract_skills_from_text(llm_evaluation)
        for skill in all_skills_in_eval:
            if skill not in matched_skills and skill not in missing_skills:
                # Check if it's mentioned positively (has, possesses, experience with, strong)
                positive_context = re.search(
                    rf'\b{re.escape(skill)}\b.*?(?:experience|mentioned|has|possesses|strong|good|excellent|extensive|proficient)',
                    llm_evaluation,
                    re.IGNORECASE
                )
                # Check if it's mentioned negatively (missing, lacks, not mentioned)
                negative_context = re.search(
                    rf'\b(?:missing|lacks?|not mentioned|not explicitly|absent|no explicit).*?\b{re.escape(skill)}\b',
                    llm_evaluation,
                    re.IGNORECASE
                )
                
                if positive_context and not negative_context:
                    matched_skills.append(skill)
                elif negative_context:
                    missing_skills.append(skill)
        
        # Remove duplicates and ensure no overlap
        matched_skills = sorted(list(set(matched_skills)))
        missing_skills = sorted(list(set([s for s in missing_skills if s not in matched_skills])))
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills
        }
    
    def analyze_skill_match(
        self,
        resume_text: str,
        job_description: str,
        llm_evaluation: str = None
    ) -> Dict[str, Any]:
        """
        Analyze skill matching between resume and job description.
        Uses LLM evaluation if available for more accurate skill extraction.
        
        Args:
            resume_text: Candidate resume text
            job_description: Job description text
            llm_evaluation: Optional LLM evaluation text for better skill extraction
            
        Returns:
            Dictionary with matched skills, missing skills, etc.
        """
        # Extract from resume and job description
        resume_skills = set(self.extract_skills_from_text(resume_text))
        job_skills = set(self.extract_skills_from_text(job_description))
        
        # If LLM evaluation is available, use it for more accurate skill extraction
        if llm_evaluation:
            llm_skills = self.extract_skills_from_llm_evaluation(llm_evaluation)
            llm_matched = set(llm_skills.get('matched_skills', []))
            llm_missing = set(llm_skills.get('missing_skills', []))
            
            # Combine regex and LLM results - LLM takes priority
            matched_skills = llm_matched if llm_matched else resume_skills.intersection(job_skills)
            missing_skills = llm_missing if llm_missing else job_skills - resume_skills
            
            # Extra skills are in resume but not required
            extra_skills = resume_skills - job_skills - matched_skills
        else:
            # Fallback to regex-only method
            matched_skills = resume_skills.intersection(job_skills)
            missing_skills = job_skills - resume_skills
            extra_skills = resume_skills - job_skills
        
        match_rate = len(matched_skills) / len(job_skills) if job_skills else 0.0
        
        return {
            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(list(missing_skills)),
            "extra_skills": sorted(list(extra_skills)),
            "match_rate": round(match_rate, 3),
            "total_job_skills": len(job_skills),
            "total_resume_skills": len(resume_skills)
        }
    
    def explain_score_breakdown(
        self,
        final_score: float,
        llm_score: float,
        semantic_similarity: float,
        semantic_weight: float = 0.3,
        llm_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        Explain how the final score was calculated.
        
        Args:
            final_score: Final hybrid score
            llm_score: LLM evaluation score
            semantic_similarity: Semantic similarity score
            semantic_weight: Weight for semantic component
            llm_weight: Weight for LLM component
            
        Returns:
            Dictionary with score breakdown explanation
        """
        semantic_contribution = semantic_weight * (semantic_similarity * 10)
        llm_contribution = llm_weight * llm_score
        
        return {
            "final_score": final_score,
            "components": {
                "semantic": {
                    "raw_score": semantic_similarity,
                    "normalized_score": round(semantic_similarity * 10, 2),
                    "weight": semantic_weight,
                    "contribution": round(semantic_contribution, 2),
                    "explanation": f"Semantic similarity found {semantic_similarity*100:.1f}% match based on keywords and skills"
                },
                "llm": {
                    "raw_score": llm_score,
                    "weight": llm_weight,
                    "contribution": round(llm_contribution, 2),
                    "explanation": f"LLM evaluation scored {llm_score}/10 based on deep analysis of fit"
                }
            },
            "formula": f"Final Score = {semantic_weight} × (Semantic × 10) + {llm_weight} × LLM Score",
            "calculation": f"{final_score} = {semantic_weight} × ({semantic_similarity} × 10) + {llm_weight} × {llm_score}"
        }
    
    def generate_explanation(
        self,
        candidate_result: Dict[str, Any],
        job_description: str,
        resume_text: str,
        vectorstore=None,
        ranker=None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a candidate ranking.
        
        Uses hybrid approach:
        - Rule-based: Skill matching, score breakdowns
        - SHAP: Feature importance (if available)
        - LIME: Text-level importance (if available)
        
        Args:
            candidate_result: Result dictionary from ranker
            job_description: Job description used
            resume_text: Full resume text
            vectorstore: Optional vectorstore for SHAP/LIME
            ranker: Optional ranker instance for LIME
            
        Returns:
            Complete explanation dictionary with SHAP and LIME insights
        """
        # Extract key insights from LLM evaluation (do this first!)
        llm_evaluation = candidate_result.get('evaluation', '')
        
        # Skill matching analysis - use LLM evaluation for better accuracy
        skill_analysis = self.analyze_skill_match(
            resume_text, 
            job_description,
            llm_evaluation=llm_evaluation
        )
        
        # Score breakdown
        score_breakdown = self.explain_score_breakdown(
            final_score=candidate_result['final_score'],
            llm_score=candidate_result['llm_score'],
            semantic_similarity=candidate_result['semantic_similarity']
        )
        
        # Try to extract structured info from LLM evaluation
        experience_match = self._extract_experience_info(llm_evaluation)
        leadership_info = self._extract_leadership_info(llm_evaluation)
        
        explanation = {
            "candidate_id": candidate_result['meta'].get('id', 'unknown'),
            "overall_score": candidate_result['final_score'],
            "score_breakdown": score_breakdown,
            "skill_analysis": skill_analysis,
            "llm_evaluation": llm_evaluation,
            "insights": {
                "experience": experience_match,
                "leadership": leadership_info,
                "strengths": self._extract_strengths(llm_evaluation),
                "weaknesses": self._extract_weaknesses(llm_evaluation)
            },
            "ranking_factors": {
                "primary": self._identify_primary_factor(score_breakdown, skill_analysis),
                "secondary": self._identify_secondary_factors(score_breakdown, skill_analysis)
            }
        }
        
        # Add SHAP explanations if available
        if self.use_shap and self.shap_explainer:
            try:
                # SHAP skill importance
                shap_skill_importance = self.shap_explainer.explain_skill_importance(
                    skills_present=skill_analysis['matched_skills'] + skill_analysis['extra_skills'],
                    skills_required=skill_analysis['matched_skills'] + skill_analysis['missing_skills'],
                    skill_weights={skill: 1.0 for skill in skill_analysis['matched_skills']}
                )
                
                explanation["shap_analysis"] = {
                    "skill_importance": shap_skill_importance,
                    "available": True
                }
                logger.debug("SHAP analysis added to explanation")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                explanation["shap_analysis"] = {"available": False, "error": str(e)}
        else:
            explanation["shap_analysis"] = {"available": False}
        
        # Add LIME explanations if available
        if self.use_lime and self.lime_explainer and ranker:
            try:
                # LIME text-level importance
                lime_explanation = self.lime_explainer.explain_resume_sections(
                    resume_text=resume_text,
                    job_description=job_description,
                    ranker=ranker,
                    num_features=15
                )
                
                explanation["lime_analysis"] = {
                    "text_importance": lime_explanation,
                    "available": True
                }
                logger.debug("LIME analysis added to explanation")
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
                explanation["lime_analysis"] = {"available": False, "error": str(e)}
        else:
            explanation["lime_analysis"] = {"available": False}
        
        return explanation
    
    def _extract_experience_info(self, evaluation_text: str) -> Dict[str, Any]:
        """Extract experience-related information from LLM evaluation."""
        # Look for years of experience
        years_match = re.search(r'(\d+)\s*(?:years?|yrs?|yr)', evaluation_text, re.IGNORECASE)
        years = int(years_match.group(1)) if years_match else None
        
        # Look for experience level
        level = None
        if re.search(r'(senior|lead|principal|architect)', evaluation_text, re.IGNORECASE):
            level = "senior"
        elif re.search(r'(junior|entry|associate)', evaluation_text, re.IGNORECASE):
            level = "junior"
        elif re.search(r'(mid|intermediate)', evaluation_text, re.IGNORECASE):
            level = "mid"
        
        return {
            "years": years,
            "level": level,
            "mentioned": years is not None or level is not None
        }
    
    def _extract_leadership_info(self, evaluation_text: str) -> Dict[str, Any]:
        """Extract leadership-related information."""
        has_leadership = bool(re.search(
            r'(leadership|leading|managed|team|mentor|supervis)', 
            evaluation_text, 
            re.IGNORECASE
        ))
        
        return {
            "has_leadership": has_leadership,
            "mentioned": has_leadership
        }
    
    def _extract_strengths(self, evaluation_text: str) -> List[str]:
        """Extract strengths mentioned in evaluation - improved extraction with full phrases."""
        strengths = []
        
        # Look for "Reasons for strong fit" or similar sections - get full sentences
        strong_fit_patterns = [
            r'\*\*Reasons? (?:for|why) (?:this candidate is a )?(?:strong|good|excellent) (?:fit|match)[:\-]?\*\*\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Reasons? for weak)[^\n]+)*)',
            r'Reasons? (?:for|why) (?:this candidate is a )?(?:strong|good|excellent) (?:fit|match)[:\-]?\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Reasons? for weak)[^\n]+)*)',
        ]
        
        for pattern in strong_fit_patterns:
            strong_fit_section = re.search(pattern, evaluation_text, re.IGNORECASE | re.MULTILINE)
            if strong_fit_section:
                strong_text = strong_fit_section.group(1)
                # Extract full bullet points or sentences
                bullets = re.findall(r'[\*\-\•]\s*([^\.\n]+)', strong_text)
                for bullet in bullets:
                    bullet = bullet.strip()
                    # Extract meaningful phrases (skip generic starts)
                    if len(bullet) > 10 and not bullet.lower().startswith(('the candidate', 'they', 'their', 'this')):
                        # Get the key part after common prefixes
                        key_part = re.sub(r'^(?:the candidate\'?s?|they|their|this|these)\s+', '', bullet, flags=re.IGNORECASE)
                        if len(key_part) > 8:
                            strengths.append(key_part.strip())
                break
        
        # Extract from "Matched skills" section - get full descriptions
        matched_section = re.search(
            r'\*\*Matched skills?:\*\*\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*)[^\n]+)*)',
            evaluation_text,
            re.IGNORECASE | re.MULTILINE
        )
        if matched_section:
            matched_text = matched_section.group(1)
            # Extract skills with context
            skill_lines = re.findall(r'[\*\-\•]\s*([^\.\n]+)', matched_text)
            for line in skill_lines:
                line = line.strip()
                # Extract skill name and description
                skill_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[a-z]+(?:\s+[a-z]+){0,2}):\s*(.+)', line)
                if skill_match:
                    skill_name, description = skill_match.groups()
                    strengths.append(f"{skill_name}: {description.strip()}")
                elif len(line) > 5:
                    strengths.append(line)
        
        # Extract positive skill mentions with full context
        positive_skill_patterns = [
            r'(?:has|possesses|strong|excellent|good|proficient|experienced|skilled|extensive|impressive)\s+(?:experience with|knowledge of|expertise in|background in)?\s*([^\.\n]+(?:python|sql|aws|java|javascript|docker|kubernetes|react|tensorflow|spark|hadoop|airflow|data engineering|machine learning)[^\.\n]*)',
            r'([^\.\n]*(?:python|sql|aws|java|javascript|docker|kubernetes|react|tensorflow|spark|hadoop|airflow|data engineering|machine learning)[^\.\n]*(?:experience|skills?|knowledge|expertise|background|proficiency))',
        ]
        
        for pattern in positive_skill_patterns:
            matches = re.findall(pattern, evaluation_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(m for m in match if m)
                match = match.strip()
                if len(match) > 8 and match not in strengths:
                    strengths.append(match)
        
        # Filter and clean
        filtered = []
        generic_starts = {'the candidate', 'they', 'their', 'this', 'these', 'the', 'a', 'an'}
        generic_words = {'fit', 'match', 'strong', 'good', 'excellent', 'candidate'}
        
        for s in strengths:
            s_lower = s.lower().strip()
            # Skip if starts with generic words or is too short
            if (len(s) > 10 and 
                not any(s_lower.startswith(g) for g in generic_starts) and
                s_lower not in generic_words and
                s not in filtered):
                # Clean up common prefixes
                cleaned = re.sub(r'^(?:the candidate\'?s?|they|their|this|these)\s+', '', s, flags=re.IGNORECASE)
                filtered.append(cleaned.strip())
        
        return filtered[:6]  # Return top 6 meaningful strengths
    
    def _extract_weaknesses(self, evaluation_text: str) -> List[str]:
        """Extract weaknesses/gaps mentioned in evaluation - improved extraction with full phrases."""
        weaknesses = []
        
        # Look for "Reasons for weak fit" or similar sections - get full sentences
        weak_fit_patterns = [
            r'\*\*Reasons? (?:for|why) (?:this candidate may be a )?(?:weak|poor|limited) (?:fit|match)[:\-]?\*\*\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Overall|Reasons? for strong)[^\n]+)*)',
            r'Reasons? (?:for|why) (?:this candidate may be a )?(?:weak|poor|limited) (?:fit|match)[:\-]?\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*|Overall|Reasons? for strong)[^\n]+)*)',
        ]
        
        for pattern in weak_fit_patterns:
            weak_fit_section = re.search(pattern, evaluation_text, re.IGNORECASE | re.MULTILINE)
            if weak_fit_section:
                weak_text = weak_fit_section.group(1)
                # Extract full bullet points or sentences
                bullets = re.findall(r'[\*\-\•]\s*([^\.\n]+)', weak_text)
                for bullet in bullets:
                    bullet = bullet.strip()
                    # Extract meaningful phrases
                    if len(bullet) > 10 and not bullet.lower().startswith(('the candidate', 'they', 'their', 'this')):
                        # Get the key part after common prefixes
                        key_part = re.sub(r'^(?:the candidate\'?s?|they|their|this|these)\s+', '', bullet, flags=re.IGNORECASE)
                        if len(key_part) > 8:
                            weaknesses.append(key_part.strip())
                break
        
        # Extract from "Missing skills" section - get full descriptions
        missing_section = re.search(
            r'\*\*Missing skills?:\*\*\s*\n?\s*(?:[\*\-\•]?\s*)?([^\n]+(?:\n(?!\*\*)[^\n]+)*)',
            evaluation_text,
            re.IGNORECASE | re.MULTILINE
        )
        if missing_section:
            missing_text = missing_section.group(1)
            # Extract skills with context
            skill_lines = re.findall(r'[\*\-\•]\s*([^\.\n]+)', missing_text)
            for line in skill_lines:
                line = line.strip()
                # Extract skill name and description
                skill_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[a-z]+(?:\s+[a-z]+){0,2}):\s*(.+)', line)
                if skill_match:
                    skill_name, description = skill_match.groups()
                    weaknesses.append(f"{skill_name}: {description.strip()}")
                elif len(line) > 5:
                    weaknesses.append(line)
        
        # Extract negative skill mentions with full context
        negative_skill_patterns = [
            r'(?:lacks?|missing|no|doesn\'t have|not mentioned|not explicitly|absent|no explicit)\s+(?:experience with|knowledge of|expertise in)?\s*([^\.\n]+(?:aws|airflow|spark|hadoop|python|sql|docker|kubernetes|data engineering)[^\.\n]*)',
            r'([^\.\n]*(?:aws|airflow|spark|hadoop|python|sql|docker|kubernetes|data engineering)[^\.\n]*(?:is not|not mentioned|not explicitly|lacks?|missing))',
        ]
        
        for pattern in negative_skill_patterns:
            matches = re.findall(pattern, evaluation_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(m for m in match if m)
                match = match.strip()
                if len(match) > 8 and match not in weaknesses:
                    weaknesses.append(match)
        
        # Filter and clean
        filtered = []
        generic_starts = {'the candidate', 'they', 'their', 'this', 'these', 'the', 'a', 'an'}
        generic_words = {'fit', 'match', 'weak', 'limited'}
        
        for w in weaknesses:
            w_lower = w.lower().strip()
            # Skip if starts with generic words or is too short
            if (len(w) > 10 and 
                not any(w_lower.startswith(g) for g in generic_starts) and
                w_lower not in generic_words and
                w not in filtered):
                # Clean up common prefixes
                cleaned = re.sub(r'^(?:the candidate\'?s?|they|their|this|these)\s+', '', w, flags=re.IGNORECASE)
                filtered.append(cleaned.strip())
        
        return filtered[:6]  # Return top 6 meaningful weaknesses
    
    def _identify_primary_factor(self, score_breakdown: Dict, skill_analysis: Dict) -> str:
        """Identify the primary factor influencing the ranking."""
        llm_contribution = score_breakdown['components']['llm']['contribution']
        semantic_contribution = score_breakdown['components']['semantic']['contribution']
        
        if llm_contribution > semantic_contribution * 2:
            return "LLM deep analysis"
        elif semantic_contribution > llm_contribution * 2:
            return "Semantic skill matching"
        elif skill_analysis['match_rate'] > 0.7:
            return "Strong skill alignment"
        else:
            return "Balanced evaluation"
    
    def _identify_secondary_factors(self, score_breakdown: Dict, skill_analysis: Dict) -> List[str]:
        """Identify secondary factors."""
        factors = []
        
        if skill_analysis['match_rate'] > 0.5:
            factors.append("Good skill overlap")
        if skill_analysis['extra_skills']:
            factors.append(f"Has additional skills: {', '.join(skill_analysis['extra_skills'][:3])}")
        if score_breakdown['components']['llm']['raw_score'] > 7:
            factors.append("Strong overall fit")
        
        return factors


def explain_ranking(
    candidate_result: Dict[str, Any],
    job_description: str,
    resume_text: str
) -> Dict[str, Any]:
    """
    Convenience function to generate explanation for a ranking.
    
    Args:
        candidate_result: Result from ranker
        job_description: Job description
        resume_text: Full resume text
        
    Returns:
        Explanation dictionary
    """
    explainer = RankingExplainer()
    return explainer.generate_explanation(candidate_result, job_description, resume_text)

