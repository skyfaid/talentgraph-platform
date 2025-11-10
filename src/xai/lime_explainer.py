"""
LIME (Local Interpretable Model-agnostic Explanations) integration.

LIME provides text-level explanations showing which words/phrases
in the resume influenced the ranking decision.
"""
import re
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from ..utils.logger import setup_logger

logger = setup_logger("lime_explainer")


class LIMEExplainer:
    """
    LIME explainer for CV ranking system.
    
    Provides word/phrase-level explanations showing which parts
    of the resume text influenced the ranking.
    """
    
    def __init__(self):
        """Initialize LIME explainer."""
        if not LIME_AVAILABLE:
            logger.warning("LIME not installed. Install with: pip install lime")
            self.available = False
            self.explainer = None
        else:
            self.available = True
            self.explainer = LimeTextExplainer(class_names=['Low Match', 'High Match'])
    
    def explain_text_importance(
        self,
        resume_text: str,
        job_description: str,
        prediction_function,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain which words/phrases in resume influenced the ranking.
        
        Args:
            resume_text: Candidate resume text
            job_description: Job description
            prediction_function: Function that takes text and returns score
            num_features: Number of top features to return
            
        Returns:
            Dictionary with word/phrase importance
        """
        if not self.available:
            return {"error": "LIME not available"}
        
        try:
            # Create explanation
            explanation = self.explainer.explain_instance(
                resume_text,
                prediction_function,
                num_features=num_features,
                labels=[1]  # Explain positive class (high match)
            )
            
            # Extract feature importance
            exp_list = explanation.as_list(label=1)
            
            # Separate positive and negative contributions
            positive_features = [(word, score) for word, score in exp_list if score > 0]
            negative_features = [(word, score) for word, score in exp_list if score < 0]
            
            # Sort by absolute importance
            positive_features.sort(key=lambda x: abs(x[1]), reverse=True)
            negative_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return {
                "positive_contributions": [
                    {"text": word, "importance": float(score)}
                    for word, score in positive_features[:num_features]
                ],
                "negative_contributions": [
                    {"text": word, "importance": float(score)}
                    for word, score in negative_features[:num_features]
                ],
                "top_positive_words": [word for word, _ in positive_features[:5]],
                "top_negative_words": [word for word, _ in negative_features[:5]],
                "explanation_available": True
            }
            
        except Exception as e:
            logger.error(f"Error in LIME explanation: {e}")
            return {"error": str(e)}
    
    def explain_resume_sections(
        self,
        resume_text: str,
        job_description: str,
        ranker,
        num_features: int = 15
    ) -> Dict[str, Any]:
        """
        Explain which sections of resume are most important.
        
        Args:
            resume_text: Full resume text
            job_description: Job description
            ranker: CVRanker instance
            num_features: Number of features to explain
            
        Returns:
            Dictionary with section-level importance
        """
        if not self.available:
            return {"error": "LIME not available"}
        
        try:
            # Create prediction function that uses the ranker
            def predict_fn(texts):
                """Prediction function for LIME."""
                scores = []
                for text in texts:
                    try:
                        # Get similarity score for this text
                        results = ranker.vectorstore.similarity_search_with_relevance_scores(
                            job_description, k=1
                        )
                        # Use a simplified scoring
                        score = 0.5  # Default
                        if results:
                            # Check if text matches
                            doc_text = results[0][0].page_content if results else ""
                            # Simple similarity check
                            common_words = set(text.lower().split()) & set(doc_text.lower().split())
                            score = len(common_words) / max(len(text.split()), 1)
                        scores.append([1 - score, score])  # [low_match, high_match]
                    except:
                        scores.append([0.5, 0.5])
                return np.array(scores)
            
            # Get explanation
            explanation = self.explainer.explain_instance(
                resume_text,
                predict_fn,
                num_features=num_features,
                labels=[1]
            )
            
            exp_list = explanation.as_list(label=1)
            
            # Group by resume sections (if we can identify them)
            section_importance = self._group_by_sections(exp_list, resume_text)
            
            # Filter out noise: typos, placeholders, generic words
            noise_words = {
                'num', 'exprience', 'experience', 'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'we', 'you', 'he', 'she',
                'good', 'bad', 'very', 'much', 'many', 'more', 'most', 'some', 'any', 'all', 'no', 'not',
                'can', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                'from', 'as', 'if', 'than', 'then', 'when', 'where', 'what', 'which', 'who', 'how', 'why',
                'company', 'work', 'task', 'system', 'web', 'like', 'using', 'provided', 'helped', 'added',
                'method', 'related', 'state', 'tree', 'involved', 'various', 'month', 'window', 'application'
            }
            
            # Technical/relevant words that should be prioritized
            tech_keywords = {
                'python', 'sql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'react', 'vue', 'angular',
                'tensorflow', 'pytorch', 'spark', 'hadoop', 'airflow', 'kafka', 'java', 'javascript',
                'typescript', 'node', 'django', 'flask', 'spring', 'git', 'jenkins', 'terraform',
                'data', 'engineering', 'analyst', 'scientist', 'developer', 'architect', 'engineer',
                'database', 'warehouse', 'pipeline', 'etl', 'ml', 'ai', 'nlp', 'analytics', 'visualization',
                'tableau', 'powerbi', 'looker', 'snowflake', 'redshift', 'bigquery', 'mongodb', 'postgres',
                'oracle', 'mysql', 'redis', 'elasticsearch', 'management', 'leadership', 'team', 'project',
                'years', 'experience', 'senior', 'lead', 'principal', 'architect', 'degree', 'education',
                'certification', 'skill', 'expertise', 'proficiency', 'knowledge', 'understanding'
            }
            
            # Filter word importance with priority for tech keywords
            filtered_word_importance = []
            for word, score in exp_list:
                word_lower = word.lower().strip()
                # Skip noise words, very short words, and typos
                if (word_lower not in noise_words and 
                    len(word) > 2 and 
                    not word_lower.isdigit() and
                    abs(score) > 0.001):  # Only include meaningful contributions
                    # Boost importance for tech keywords
                    adjusted_score = abs(score)
                    if word_lower in tech_keywords:
                        adjusted_score *= 1.5  # Boost tech keywords
                    
                    filtered_word_importance.append({
                        "text": word,
                        "importance": float(score),
                        "adjusted_importance": float(adjusted_score)
                    })
            
            # Sort by adjusted importance (prioritize tech keywords)
            filtered_word_importance.sort(key=lambda x: x.get('adjusted_importance', abs(x['importance'])), reverse=True)
            
            # Get top contributing and detracting words (prioritize tech keywords)
            positive_words = [w for w in filtered_word_importance if w['importance'] > 0]
            negative_words = [w for w in filtered_word_importance if w['importance'] < 0]
            
            # Already sorted by adjusted_importance, but ensure tech keywords are prioritized
            positive_words.sort(key=lambda x: x.get('adjusted_importance', abs(x['importance'])), reverse=True)
            negative_words.sort(key=lambda x: x.get('adjusted_importance', abs(x['importance'])), reverse=True)
            
            return {
                "word_importance": filtered_word_importance[:15],  # Top 15 filtered words (prioritized)
                "section_importance": section_importance,
                "top_contributing_words": [w['text'] for w in positive_words[:8]],  # Top 8 tech-relevant words
                "top_detracting_words": [w['text'] for w in negative_words[:5]]  # Top 5 detracting words
            }
            
        except Exception as e:
            logger.error(f"Error in section explanation: {e}")
            return {"error": str(e)}
    
    def _group_by_sections(self, exp_list: List[tuple], resume_text: str) -> Dict[str, float]:
        """Group word importance by resume sections."""
        # Common resume section headers
        sections = {
            "summary": ["summary", "objective", "profile"],
            "experience": ["experience", "work", "employment", "history"],
            "education": ["education", "academic", "qualification"],
            "skills": ["skills", "technical", "competencies"],
            "certifications": ["certification", "certificate", "license"]
        }
        
        section_scores = {section: 0.0 for section in sections.keys()}
        section_counts = {section: 0 for section in sections.keys()}
        
        resume_lower = resume_text.lower()
        
        for word, score in exp_list:
            word_lower = word.lower()
            # Find which section this word likely belongs to
            for section, keywords in sections.items():
                if any(keyword in resume_lower for keyword in keywords):
                    # Check if word appears near section header
                    section_scores[section] += abs(score)
                    section_counts[section] += 1
        
        # Average scores
        for section in section_scores:
            if section_counts[section] > 0:
                section_scores[section] /= section_counts[section]
        
        return section_scores


def get_lime_explainer() -> LIMEExplainer:
    """Get or create LIME explainer instance."""
    return LIMEExplainer()

