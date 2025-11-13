"""
Combined Text Analysis Module
Handles sentiment analysis and NER extraction
"""
import re
import spacy
from transformers import pipeline
from typing import Dict, List
from datetime import datetime, timedelta

from utils.config import SENTIMENT_MODEL, MOTIVATION_KEYWORDS, RED_FLAG_KEYWORDS, SKILL_KEYWORDS, MIN_ANSWER_LENGTH, RED_FLAG_SEVERITY


class TextAnalyzer:
    def __init__(self):
        """Initialize sentiment and NER models"""
        print("Loading sentiment model...")
        self.sentiment_analyzer = pipeline("text-classification", model=SENTIMENT_MODEL, top_k=None)
        
        print("Loading spaCy NER model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment/emotion from text
        
        Returns:
            Dictionary of emotions with scores
        """
        try:
            results = self.sentiment_analyzer(text)[0]
            emotions = {item['label']: item['score'] for item in results}
            
            # Calculate motivation score based on positive emotions
            motivation_score = (
                emotions.get('joy', 0) * 0.4 +
                emotions.get('surprise', 0) * 0.2 +
                emotions.get('neutral', 0) * 0.2 +
                (1 - emotions.get('sadness', 0)) * 0.1 +
                (1 - emotions.get('anger', 0)) * 0.1
            ) * 10
            
            emotions['motivation_score'] = min(10, max(1, motivation_score))
            return emotions
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {'motivation_score': 5.0}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (skills, dates, companies, etc.)
        
        Returns:
            Dictionary of entity types and values
        """
        try:
            doc = self.nlp(text)
            
            entities = {
                'organizations': [],
                'dates': [],
                'skills': [],
                'locations': []
            }
            
            # Extract spaCy entities
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'GPE':
                    entities['locations'].append(ent.text)
            
            # Extract skills via keyword matching
            text_lower = text.lower()
            entities['skills'] = [skill for skill in SKILL_KEYWORDS if skill in text_lower]
            
            return entities
            
        except Exception as e:
            print(f"Error in NER: {e}")
            return {'organizations': [], 'dates': [], 'skills': [], 'locations': []}
    
    def extract_availability(self, text: str) -> Dict[str, any]:
        """
        Extract availability information (notice period, start date)
        
        Returns:
            Availability details
        """
        text_lower = text.lower()
        
        result = {
            'notice_period': None,
            'available_immediately': False,
            'start_date_mentioned': False
        }
        
        # Check for immediate availability
        if any(phrase in text_lower for phrase in ['immediate', 'right away', 'asap', 'immediately']):
            result['available_immediately'] = True
            result['notice_period'] = '0 days'
            return result
        
        # Extract notice period patterns
        patterns = [
            (r'(\d+)\s*weeks?\s*notice', 'weeks'),
            (r'(\d+)\s*months?\s*notice', 'months'),
            (r'(\d+)\s*days?\s*notice', 'days'),
            (r'two\s*weeks?', '2 weeks'),
            (r'one\s*month', '1 month')
        ]
        
        for pattern, unit in patterns:
            match = re.search(pattern, text_lower)
            if match:
                if isinstance(unit, str) and 'weeks' in unit:
                    result['notice_period'] = f"{match.group(1)} weeks"
                elif isinstance(unit, str) and 'months' in unit:
                    result['notice_period'] = f"{match.group(1)} months"
                elif isinstance(unit, str) and 'days' in unit:
                    result['notice_period'] = f"{match.group(1)} days"
                else:
                    result['notice_period'] = unit
                break
        
        # Check if specific start date mentioned
        if any(word in text_lower for word in ['start', 'begin', 'join']):
            result['start_date_mentioned'] = True
        
        return result
    
    def extract_salary_info(self, text: str) -> Dict[str, any]:
        """
        Extract salary expectations
        
        Returns:
            Salary information
        """
        result = {
            'salary_mentioned': False,
            'amount': None,
            'negotiable': False
        }
        
        text_lower = text.lower()
        
        # Check if negotiable
        if any(word in text_lower for word in ['negotiable', 'flexible', 'open to discuss']):
            result['negotiable'] = True
        
        # Extract salary amounts (basic patterns)
        patterns = [
            r'[\$€£]\s*(\d+(?:,\d{3})*(?:k)?)',
            r'(\d+(?:,\d{3})*)\s*(?:dollars|euros|pounds)',
            r'(\d+)k'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                result['salary_mentioned'] = True
                result['amount'] = match.group(1)
                break
        
        return result
    
    def calculate_motivation_score(self, text: str, sentiment_score: float) -> float:
        """
        Calculate motivation score from text + sentiment
        
        Returns:
            Motivation score (1-10)
        """
        text_lower = text.lower()

        # Count motivation keywords
        motivation_count = sum(1 for keyword in MOTIVATION_KEYWORDS if keyword in text_lower)

        # Count red flag keywords (negative impact)
        red_flag_count = sum(1 for keyword in RED_FLAG_KEYWORDS if keyword in text_lower)

        # Combine scores
        keyword_score = min(10, 5 + motivation_count - red_flag_count * 2)

        # Weighted combination
        final_score = (sentiment_score * 0.6) + (keyword_score * 0.4)

        return min(10, max(1, final_score))
    
    def calculate_skill_match(self, mentioned_skills: List[str], required_skills: List[str]) -> Dict[str, any]:
        """
        Calculate skill matching score
        
        Args:
            mentioned_skills: Skills extracted from candidate answers
            required_skills: Required skills for the job
            
        Returns:
            Skill match analysis
        """
        if not required_skills:
            return {
                'match_score': 0,
                'matched_skills': [],
                'missing_skills': [],
                'match_percentage': 0
            }
        
        # Normalize for comparison
        mentioned_lower = [s.lower().strip() for s in mentioned_skills]
        required_lower = [s.lower().strip() for s in required_skills]
        
        # Find matches
        matched = [skill for skill in required_lower if skill in mentioned_lower]
        missing = [skill for skill in required_lower if skill not in mentioned_lower]
        
        match_percentage = (len(matched) / len(required_lower)) * 100 if required_lower else 0
        
        # Calculate match score (0-10 scale)
        match_score = min(10, (len(matched) / max(1, len(required_lower))) * 10)
        
        return {
            'match_score': round(match_score, 1),
            'matched_skills': matched,
            'missing_skills': missing,
            'match_percentage': round(match_percentage, 1)
        }

    def detect_red_flags(self, text: str) -> List[dict]:
        """
        Detect red flags in candidate responses with severity levels.

        Returns:
            List of dictionaries with flag details and severity
        """
        text_lower = text.lower()
        detected = []

        # Check flat red flag keywords (all default to 'major' severity)
        for keyword in RED_FLAG_KEYWORDS:
            if keyword in text_lower:
                detected.append({
                    'phrase': keyword,
                    'severity': 'major',
                    'penalty': RED_FLAG_SEVERITY.get('major', 0)
                })

        # Check for very short answers (context-dependent)
        word_count = len(text.split())
        if word_count < MIN_ANSWER_LENGTH and word_count > 1:  # Ignore single-word answers
            detected.append({
                'phrase': f"Brief answer ({word_count} words)",
                'severity': 'minor',
                'penalty': RED_FLAG_SEVERITY.get('minor', 0)
            })
        elif word_count == 1:
            # Single word answers are more concerning
            detected.append({
                'phrase': "Single-word answer",
                'severity': 'major',
                'penalty': RED_FLAG_SEVERITY.get('major', 0)
            })

        # Check for negative company mentions
        negative_patterns = [
            (r'hate(?:d)?\s+(?:my|the)', 'major'),
            (r'terrible\s+(?:boss|manager|company)', 'major'),
            (r'worst\s+(?:job|company|experience)', 'major'),
        ]

        for pattern, severity in negative_patterns:
            if re.search(pattern, text_lower):
                detected.append({
                    'phrase': "Negative language about previous employer",
                    'severity': severity,
                    'penalty': RED_FLAG_SEVERITY.get(severity, 0)
                })
                break

        return detected
