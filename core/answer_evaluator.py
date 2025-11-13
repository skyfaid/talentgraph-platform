"""
Answer Evaluation Module using Mistral 7B
"""
import requests
import json
import re
from typing import Dict, Tuple

from utils.config import MISTRAL_MODEL


class AnswerEvaluator:
    def __init__(self):
        """Initialize Mistral connection"""
        self.model = MISTRAL_MODEL
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def evaluate_answer(
        self, 
        question: str, 
        answer: str,
        job_context: Dict = None
    ) -> Dict[str, any]:
        """
        Evaluate candidate's answer quality
        
        Args:
            question: The question asked
            answer: Candidate's answer
            job_context: Job requirements context
            
        Returns:
            Dictionary with score and analysis
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(question, answer, job_context)
        
        # Get Mistral's evaluation
        evaluation_text = self._call_mistral(prompt)
        
        # Parse the response
        result = self._parse_evaluation(evaluation_text, answer)
        
        return result
    
    def _build_evaluation_prompt(
        self, 
        question: str, 
        answer: str,
        job_context: Dict = None
    ) -> str:
        """Build prompt for answer evaluation"""
        
        job_info = ""
        if job_context:
            job_info = f"\nJob Context:\n- Role: {job_context.get('title', 'N/A')}\n- Key Skills: {', '.join(job_context.get('skills', []))}\n"
        
        prompt = f"""You are an expert recruiter evaluating a candidate's interview answer.

Question: "{question}"
Candidate's Answer: "{answer}"
{job_info}

Evaluate this answer on a scale of 1-10 based on:
1. Relevance - Does it address the question directly?
2. Depth - Is it specific with examples, or vague/generic?
3. Professionalism - Is the tone appropriate for a job interview?
4. Red Flags - Any concerning statements or attitudes?

Scoring guidelines:
- 8-10: Excellent answer with specifics, enthusiasm, and relevance
- 6-7: Good answer but could be more detailed
- 4-5: Generic or somewhat vague answer
- 1-3: Poor answer, off-topic, or concerning

Provide your evaluation in this EXACT format:
SCORE: [number 1-10]
RELEVANCE: [1-2 sentences explaining relevance]
DEPTH: [1-2 sentences on specificity and detail]
RED_FLAGS: [list specific concerns, or "None"]
OVERALL: [2-3 sentences summary and justification of score]

Be fair but objective. Consider that brief answers aren't always bad if they directly answer the question."""
        
        return prompt
    
    def _call_mistral(self, prompt: str) -> str:
        """Call Mistral API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temp for more consistent evaluation
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"Mistral API error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error calling Mistral: {e}")
            return ""
    
    def _parse_evaluation(self, evaluation_text: str, answer: str) -> Dict:
        """Parse Mistral's evaluation response"""
        result = {
            "score": 5.0,
            "relevance": "",
            "depth": "",
            "red_flags": [],
            "overall": "",
            "answer_length": len(answer.split())
        }
        
        if not evaluation_text:
            return result
        
        try:
            # Extract score
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
            if score_match:
                result["score"] = float(score_match.group(1))
            
            # Extract relevance
            relevance_match = re.search(r'RELEVANCE:\s*(.+?)(?=DEPTH:|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            if relevance_match:
                result["relevance"] = relevance_match.group(1).strip()
            
            # Extract depth
            depth_match = re.search(r'DEPTH:\s*(.+?)(?=RED_FLAGS:|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            if depth_match:
                result["depth"] = depth_match.group(1).strip()
            
            # Extract red flags
            red_flags_match = re.search(r'RED_FLAGS:\s*(.+?)(?=OVERALL:|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            if red_flags_match:
                flags_text = red_flags_match.group(1).strip()
                if flags_text.lower() != "none" and flags_text != "":
                    result["red_flags"] = [f.strip() for f in flags_text.split('\n') if f.strip()]
            
            # Extract overall
            overall_match = re.search(r'OVERALL:\s*(.+?)$', evaluation_text, re.IGNORECASE | re.DOTALL)
            if overall_match:
                result["overall"] = overall_match.group(1).strip()
                
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
        
        return result
    
    def quick_quality_check(self, answer: str) -> Dict[str, any]:
        """
        Quick heuristic-based quality check (no LLM)
        
        Args:
            answer: Candidate's answer
            
        Returns:
            Quality metrics
        """
        words = answer.split()
        word_count = len(words)
        
        # Check answer length
        too_short = word_count < 10
        too_long = word_count > 200
        
        # Check for filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'i mean', 'basically', 'literally']
        filler_count = sum(answer.lower().count(filler) for filler in filler_words)
        
        # Check for vague language
        vague_phrases = ['i think', 'maybe', 'i guess', 'probably', 'kind of', 'sort of']
        vague_count = sum(answer.lower().count(phrase) for phrase in vague_phrases)
        
        return {
            "word_count": word_count,
            "too_short": too_short,
            "too_long": too_long,
            "filler_count": filler_count,
            "vague_count": vague_count,
            "quality_score": max(1, 10 - too_short*3 - too_long*2 - filler_count - vague_count)
        }
