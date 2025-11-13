"""
Enhanced Scoring System with XAI Integration
"""
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import SCORING_WEIGHTS, INTERVIEW_LOGS_PATH


class ScoringSystem:
    def __init__(self):
        """Initialize scoring system with XAI"""
        self.weights = SCORING_WEIGHTS
        
        # Import XAI explainer
        try:
            from xai_explainer import XAIExplainer
            self.xai = XAIExplainer()
        except ImportError:
            print("Warning: XAI explainer not found. Running without explanations.")
            self.xai = None
    
    def calculate_final_score(self, interview_data: Dict) -> Dict[str, any]:
        """
        Calculate final weighted score with XAI explanations
        
        Args:
            interview_data: Dictionary with all analysis results
            
        Returns:
            Final scores with explanations and XAI insights
        """
        scores = {
            'motivation': 0,
            'experience': 0,
            'logistics': 0,
            'red_flags': 0,
            'confidence': 0
        }
        
        explanations = {}
        
        # 1. Motivation Score
        text_motivation = interview_data.get('text_sentiment', {}).get('motivation_score', 5)
        audio_confidence = interview_data.get('audio_emotion', {}).get('confidence_score', 5)
        video_emotion = interview_data.get('video_emotion', {})
        
        if video_emotion:
            video_positive = video_emotion.get('happy', 0) + video_emotion.get('neutral', 0)
            video_score = min(10, video_positive * 10)
            scores['motivation'] = (text_motivation * 0.4 + audio_confidence * 0.3 + video_score * 0.3)
            explanations['motivation'] = (
                f"Text sentiment: {text_motivation:.1f}/10, "
                f"Voice confidence: {audio_confidence:.1f}/10, "
                f"Video positivity: {video_score:.1f}/10"
            )
        else:
            scores['motivation'] = (text_motivation * 0.6 + audio_confidence * 0.4)
            explanations['motivation'] = (
                f"Text sentiment: {text_motivation:.1f}/10, "
                f"Voice confidence: {audio_confidence:.1f}/10"
            )
        
        # 2. Experience Score
        answer_evaluations = interview_data.get('answer_evaluations', [])
        if answer_evaluations:
            avg_answer_score = sum(a['score'] for a in answer_evaluations) / len(answer_evaluations)
            relevance_scores = [a.get('relevance_score', 5) for a in answer_evaluations]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 5
            relevance_factor = avg_relevance / 10.0
            avg_answer_score = avg_answer_score * relevance_factor
        else:
            avg_answer_score = 5
            avg_relevance = 5
        
        skill_match = interview_data.get('skill_match', {})
        skill_bonus = skill_match.get('match_score', 0) * 0.2
        
        scores['experience'] = min(10, avg_answer_score + skill_bonus)
        
        matched_count = len(skill_match.get('matched_skills', []))
        required_count = matched_count + len(skill_match.get('missing_skills', []))
        explanations['experience'] = (
            f"Answer quality: {avg_answer_score:.1f}/10, "
            f"Skills: {matched_count}/{required_count} matched"
        )
        
        # 3. Logistics Score
        availability = interview_data.get('availability', {})
        if availability.get('available_immediately'):
            scores['logistics'] = 10
            explanations['logistics'] = "Immediate availability"
        elif availability.get('notice_period'):
            scores['logistics'] = 8
            explanations['logistics'] = f"Notice: {availability['notice_period']}"
        else:
            scores['logistics'] = 5
            explanations['logistics'] = "Availability unclear"
        
        # 4. Red Flags
        red_flags = interview_data.get('red_flags', [])
        total_penalty = sum(flag.get('penalty', 1.0) for flag in red_flags)
        scores['red_flags'] = max(0, 10 - total_penalty)
        
        if red_flags:
            critical = sum(1 for f in red_flags if f.get('severity') == 'critical')
            major = sum(1 for f in red_flags if f.get('severity') == 'major')
            minor = sum(1 for f in red_flags if f.get('severity') == 'minor')
            explanations['red_flags'] = (
                f"{len(red_flags)} flags: {critical} critical, {major} major, {minor} minor"
            )
        else:
            explanations['red_flags'] = "No concerns detected"
        
        # 5. Confidence Score
        if video_emotion:
            video_conf = (video_emotion.get('happy', 0) + video_emotion.get('neutral', 0)) * 10
            scores['confidence'] = (audio_confidence * 0.5 + video_conf * 0.5)
            explanations['confidence'] = "Voice + facial analysis"
        else:
            scores['confidence'] = audio_confidence
            explanations['confidence'] = "Voice analysis only"
        
        # Calculate weighted final score
        final_score = sum(scores[key] * self.weights[key] for key in scores.keys())
        final_score = round(final_score, 2)
        
        # Generate XAI explanations
        xai_explanations = {}
        if self.xai:
            # Audio emotion explanation
            if interview_data.get('audio_emotion'):
                audio_features = interview_data.get('audio_features', {})
                xai_explanations['audio'] = self.xai.explain_audio_emotion_prediction(
                    audio_features,
                    interview_data.get('audio_emotion', {})
                )
            
            # Video emotion explanation
            if interview_data.get('video_emotion'):
                xai_explanations['video'] = self.xai.explain_video_emotion_prediction(
                    interview_data.get('video_emotion', {}),
                    interview_data.get('video_timeline', [])
                )
            
            # Score breakdown explanation
            xai_explanations['score'] = self.xai.explain_final_score(
                scores,
                self.weights,
                final_score
            )
        
        return {
            'final_score': final_score,
            'component_scores': scores,
            'explanations': explanations,
            'xai_explanations': xai_explanations,
            'recommendation': self._get_recommendation(final_score, red_flags)
        }
    
    def _get_recommendation(self, score: float, red_flags: List) -> str:
        """Generate hiring recommendation"""
        critical_flags = sum(1 for flag in red_flags if flag.get('severity') == 'critical')
        major_flags = sum(1 for flag in red_flags if flag.get('severity') == 'major')
        
        if critical_flags > 0:
            return "Reject - Critical red flags detected"
        elif major_flags >= 3:
            return "Reject - Multiple major concerns"
        elif score >= 8.0:
            return "Strong Yes - Proceed immediately"
        elif score >= 6.5:
            return "Yes - Proceed to next round"
        elif score >= 5.0:
            return "Maybe - Consider for future"
        else:
            return "No - Does not meet requirements"
    
    def generate_report(self, interview_data: Dict, scores: Dict) -> Dict:
        """Generate comprehensive report with XAI insights"""
        availability = interview_data.get('availability', {})
        entities = interview_data.get('extracted_entities', {})
        
        availability_text = "Not specified"
        if availability.get('available_immediately'):
            availability_text = "Immediate"
        elif availability.get('notice_period'):
            availability_text = availability['notice_period']
        
        insights = {
            'availability': availability_text,
            'skills_mentioned': entities.get('skills', []),
            'previous_companies': entities.get('organizations', []),
            'motivation_level': self._get_level(scores['component_scores']['motivation']),
            'concerns': self._format_red_flags(interview_data.get('red_flags', []))
        }
        
        summary = self._generate_summary(interview_data, scores)
        
        # Generate feedback using XAI
        feedback = None
        if self.xai:
            feedback = self.xai.generate_feedback_report(
                interview_data,
                scores,
                scores.get('xai_explanations', {}).get('audio'),
                scores.get('xai_explanations', {}).get('video'),
                scores.get('xai_explanations', {}).get('score')
            )
        
        report = {
            'metadata': {
                'candidate_name': interview_data.get('candidate_name', 'Unknown'),
                'interview_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': interview_data.get('duration', 0),
                'num_questions': len(interview_data.get('qa_pairs', []))
            },
            'scores': {
                'overall_score': scores['final_score'],
                'motivation': scores['component_scores']['motivation'],
                'experience_fit': scores['component_scores']['experience'],
                'logistics': scores['component_scores']['logistics'],
                'confidence': scores['component_scores']['confidence'],
                'red_flags_score': scores['component_scores']['red_flags']
            },
            'key_insights': insights,
            'summary': summary,
            'recommendation': scores['recommendation'],
            'xai_explanations': scores.get('xai_explanations', {}),
            'feedback': feedback,
            'transcript': interview_data.get('qa_pairs', []),
            'detailed_analysis': {
                'sentiment': interview_data.get('text_sentiment', {}),
                'audio_emotion': interview_data.get('audio_emotion', {}),
                'video_emotion': interview_data.get('video_emotion', {}),
                'answer_evaluations': interview_data.get('answer_evaluations', [])
            }
        }
        
        return report
    
    def _format_red_flags(self, red_flags: List[Dict]) -> List[str]:
        """Convert red flag dictionaries to strings"""
        if not red_flags:
            return []
        
        formatted = []
        for flag in red_flags:
            if isinstance(flag, dict):
                phrase = flag.get('phrase', 'Unknown')
                severity = flag.get('severity', 'unknown')
                formatted.append(f"{phrase} ({severity})")
            else:
                formatted.append(str(flag))
        
        return formatted
    
    def _generate_summary(self, interview_data: Dict, scores: Dict) -> str:
        """Generate summary"""
        motivation_level = self._get_level(scores['component_scores']['motivation'])
        experience_level = self._get_level(scores['component_scores']['experience'])
        
        skills = interview_data.get('extracted_entities', {}).get('skills', [])
        skills_text = f"Mentioned: {', '.join(skills[:5])}" if skills else "No specific skills"
        
        red_flags = interview_data.get('red_flags', [])
        if red_flags:
            flag_phrases = [
                flag.get('phrase', 'Unknown') if isinstance(flag, dict) else str(flag)
                for flag in red_flags[:3]
            ]
            concerns_text = f"Concerns: {'; '.join(flag_phrases)}"
        else:
            concerns_text = "No major concerns"
        
        return (
            f"Candidate showed {motivation_level} motivation and {experience_level} experience. "
            f"{skills_text}. {concerns_text}."
        )
    
    def _get_level(self, score: float) -> str:
        """Convert score to level"""
        if score >= 8:
            return "high"
        elif score >= 6:
            return "moderate"
        else:
            return "low"
    
    def save_report(self, report: Dict, filename: str = None):
        """Save report"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            candidate = report['metadata']['candidate_name'].replace(' ', '_')
            filename = f"interview_{candidate}_{timestamp}.json"
        
        filepath = INTERVIEW_LOGS_PATH / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {filepath}")
        return filepath