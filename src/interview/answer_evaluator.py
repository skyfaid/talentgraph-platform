"""
Answer Evaluator for AI Interview System.

Evaluates candidate answers and provides:
- Score (0-10)
- Strengths and weaknesses
- Key points extracted
- Comparison against job requirements
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_groq import ChatGroq
import re

from ..llm.groq_service import initialize_llm
from ..utils.logger import setup_logger
from ..utils.text_utils import extract_score

logger = setup_logger("answer_evaluator")


class AnswerEvaluator:
    """
    Evaluates interview answers using LLM.
    
    Supports evaluation of:
    - Technical answers (accuracy, depth, problem-solving)
    - Behavioral answers (STAR format, communication, relevance)
    """
    
    def __init__(self, llm: Optional[ChatGroq] = None):
        """
        Initialize answer evaluator.
        
        Args:
            llm: Optional LLM instance. If None, creates new one.
        """
        if llm is None:
            self.llm = initialize_llm()
        else:
            self.llm = llm
        
        self._technical_chain = None
        self._behavioral_chain = None
        
        logger.info("AnswerEvaluator initialized")
    
    def _get_technical_evaluation_chain(self) -> LLMChain:
        """Get or create technical answer evaluation chain."""
        if self._technical_chain is None:
            prompt = PromptTemplate(
                input_variables=["question", "answer", "expected_points", "job_description", "skills_tested"],
                template=(
                    "You are an expert technical interviewer evaluating a candidate's answer.\n\n"
                    "QUESTION:\n{question}\n\n"
                    "CANDIDATE ANSWER:\n{answer}\n\n"
                    "EXPECTED POINTS:\n{expected_points}\n\n"
                    "SKILLS TESTED:\n{skills_tested}\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "Evaluate the answer on:\n"
                    "1. Technical accuracy (40%) - Is the answer technically correct?\n"
                    "2. Depth of knowledge (30%) - Does it show deep understanding?\n"
                    "3. Problem-solving approach (20%) - Is the approach logical?\n"
                    "4. Communication clarity (10%) - Is it well-explained?\n\n"
                    "Provide evaluation in this format:\n"
                    "SCORE: [0-10]\n"
                    "STRENGTHS:\n- [strength 1]\n- [strength 2]\n"
                    "WEAKNESSES:\n- [weakness 1]\n- [weakness 2]\n"
                    "KEY_POINTS:\n- [key point 1]\n- [key point 2]\n"
                    "RECOMMENDATION: [strong/weak/needs_followup]\n"
                    "FEEDBACK: [detailed feedback text]\n\n"
                    "Evaluate now:"
                )
            )
            self._technical_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key="evaluation"
            )
        return self._technical_chain
    
    def _get_behavioral_evaluation_chain(self) -> LLMChain:
        """Get or create behavioral answer evaluation chain."""
        if self._behavioral_chain is None:
            prompt = PromptTemplate(
                input_variables=["question", "answer", "star_required", "job_description", "skills_assessed"],
                template=(
                    "You are an expert HR interviewer evaluating a behavioral answer in STAR format.\n\n"
                    "QUESTION:\n{question}\n\n"
                    "CANDIDATE ANSWER:\n{answer}\n\n"
                    "STAR REQUIREMENTS:\n{star_required}\n\n"
                    "SKILLS ASSESSED:\n{skills_assessed}\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "Evaluate the answer on:\n"
                    "1. STAR completeness (30%) - Does it include Situation, Task, Action, Result?\n"
                    "2. Relevance to question (25%) - Does it answer what was asked?\n"
                    "3. Demonstrated skills (25%) - Does it show the skills being tested?\n"
                    "4. Communication quality (20%) - Is it clear and well-structured?\n\n"
                    "Provide evaluation in this format:\n"
                    "SCORE: [0-10]\n"
                    "STAR_COMPLETENESS: [complete/partial/missing]\n"
                    "STRENGTHS:\n- [strength 1]\n- [strength 2]\n"
                    "WEAKNESSES:\n- [weakness 1]\n- [weakness 2]\n"
                    "KEY_POINTS:\n- [key point 1]\n- [key point 2]\n"
                    "RECOMMENDATION: [strong/weak/needs_followup]\n"
                    "FEEDBACK: [detailed feedback text]\n\n"
                    "Evaluate now:"
                )
            )
            self._behavioral_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key="evaluation"
            )
        return self._behavioral_chain
    
    def evaluate_answer(
        self,
        question: Dict[str, Any],
        answer: str,
        job_description: str,
        question_type: str = "technical"
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate's answer.
        
        Args:
            question: Question dictionary with metadata
            answer: Candidate's answer text
            job_description: Job description for context
            question_type: "technical" or "behavioral"
        
        Returns:
            Evaluation dictionary with:
            - score: 0-10
            - strengths: List of strengths
            - weaknesses: List of weaknesses
            - key_points: Extracted key points
            - recommendation: strong/weak/needs_followup
            - feedback: Detailed feedback text
            - star_completeness: (for behavioral) complete/partial/missing
        """
        try:
            if question_type == "behavioral":
                chain = self._get_behavioral_evaluation_chain()
                result = chain.invoke({
                    "question": question.get("question", ""),
                    "answer": answer,
                    "star_required": question.get("star_required", ""),
                    "job_description": job_description,
                    "skills_assessed": ", ".join(question.get("skills_assessed", []))
                })
            else:
                chain = self._get_technical_evaluation_chain()
                result = chain.invoke({
                    "question": question.get("question", ""),
                    "answer": answer,
                    "expected_points": question.get("expected_points", ""),
                    "job_description": job_description,
                    "skills_tested": ", ".join(question.get("skills_tested", []))
                })
            
            evaluation_text = result.get("evaluation", result.get("analysis", ""))
            evaluation = self._parse_evaluation(evaluation_text, question_type)
            
            logger.info(f"Evaluated {question_type} answer, score: {evaluation.get('score', 0)}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return {
                "score": 0.0,
                "strengths": [],
                "weaknesses": ["Evaluation error occurred"],
                "key_points": [],
                "recommendation": "needs_followup",
                "feedback": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def _parse_evaluation(self, text: str, question_type: str) -> Dict[str, Any]:
        """Parse evaluation from LLM output."""
        evaluation = {
            "score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "key_points": [],
            "recommendation": "needs_followup",
            "feedback": ""
        }
        
        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            evaluation["score"] = float(score_match.group(1))
        else:
            # Try to extract score using existing utility
            evaluation["score"] = extract_score(text)
        
        # Extract STAR completeness (for behavioral)
        if question_type == "behavioral":
            star_match = re.search(r'STAR_COMPLETENESS:\s*(\w+)', text, re.IGNORECASE)
            if star_match:
                evaluation["star_completeness"] = star_match.group(1).lower()
        
        # Extract strengths
        if "STRENGTHS:" in text:
            strengths_start = text.find("STRENGTHS:") + len("STRENGTHS:")
            strengths_end = text.find("WEAKNESSES:", strengths_start)
            if strengths_end == -1:
                strengths_end = text.find("KEY_POINTS:", strengths_start)
            strengths_text = text[strengths_start:strengths_end].strip() if strengths_end > 0 else text[strengths_start:].strip()
            evaluation["strengths"] = self._parse_list_items(strengths_text)
        
        # Extract weaknesses
        if "WEAKNESSES:" in text:
            weaknesses_start = text.find("WEAKNESSES:") + len("WEAKNESSES:")
            weaknesses_end = text.find("KEY_POINTS:", weaknesses_start)
            if weaknesses_end == -1:
                weaknesses_end = text.find("RECOMMENDATION:", weaknesses_start)
            weaknesses_text = text[weaknesses_start:weaknesses_end].strip() if weaknesses_end > 0 else text[weaknesses_start:].strip()
            evaluation["weaknesses"] = self._parse_list_items(weaknesses_text)
        
        # Extract key points
        if "KEY_POINTS:" in text:
            kp_start = text.find("KEY_POINTS:") + len("KEY_POINTS:")
            kp_end = text.find("RECOMMENDATION:", kp_start)
            if kp_end == -1:
                kp_end = text.find("FEEDBACK:", kp_start)
            kp_text = text[kp_start:kp_end].strip() if kp_end > 0 else text[kp_start:].strip()
            evaluation["key_points"] = self._parse_list_items(kp_text)
        
        # Extract recommendation
        rec_match = re.search(r'RECOMMENDATION:\s*(\w+)', text, re.IGNORECASE)
        if rec_match:
            evaluation["recommendation"] = rec_match.group(1).lower()
        
        # Extract feedback
        if "FEEDBACK:" in text:
            feedback_start = text.find("FEEDBACK:") + len("FEEDBACK:")
            evaluation["feedback"] = text[feedback_start:].strip()
        else:
            # Use full text as feedback if no explicit section
            evaluation["feedback"] = text.strip()
        
        return evaluation
    
    def _parse_list_items(self, text: str) -> list:
        """Parse bullet points or list items from text."""
        items = []
        # Split by newlines and dashes
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove bullet markers
            line = re.sub(r'^[\*\-\•]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            if line:
                items.append(line)
        return items


# Singleton instance
_answer_evaluator = None


def get_answer_evaluator(llm=None) -> AnswerEvaluator:
    """
    Get or create answer evaluator instance (singleton).
    
    Args:
        llm: Optional LLM instance
    
    Returns:
        AnswerEvaluator instance
    """
    global _answer_evaluator
    if _answer_evaluator is None:
        _answer_evaluator = AnswerEvaluator(llm=llm)
    return _answer_evaluator

