"""
Question Generator for AI Interview System.

Generates contextual interview questions based on:
- Job description (required skills, experience level)
- Candidate resume (their skills, experience)
- Question type (technical, behavioral, follow-up)

Uses LLM to generate relevant, role-specific questions.
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_groq import ChatGroq

from ..llm.groq_service import initialize_llm
from ..utils.logger import setup_logger

logger = setup_logger("question_generator")


class QuestionGenerator:
    """
    Generates interview questions using LLM.
    
    Supports:
    - Technical questions (based on job requirements)
    - Behavioral questions (STAR format)
    - Follow-up questions (based on previous answers)
    """
    
    def __init__(self, llm: Optional[ChatGroq] = None):
        """
        Initialize question generator.
        
        Args:
            llm: Optional LLM instance. If None, creates new one.
        """
        if llm is None:
            self.llm = initialize_llm()
        else:
            self.llm = llm
        
        # Create chains for different question types
        self._technical_chain = None
        self._behavioral_chain = None
        self._followup_chain = None
        
        logger.info("QuestionGenerator initialized")
    
    def _get_technical_chain(self) -> LLMChain:
        """Get or create technical question generation chain."""
        if self._technical_chain is None:
            prompt = PromptTemplate(
                input_variables=["job_description", "candidate_resume", "num_questions"],
                template=(
                    "You are an expert technical interviewer creating interview questions.\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "CANDIDATE RESUME:\n{candidate_resume}\n\n"
                    "Generate {num_questions} technical interview questions that:\n"
                    "1. Test the required skills mentioned in the job description\n"
                    "2. Match the candidate's experience level\n"
                    "3. Are relevant to the role\n"
                    "4. Include expected answer guidelines\n\n"
                    "Format each question as:\n"
                    "QUESTION: [question text]\n"
                    "EXPECTED_POINTS: [key points to look for in answer]\n"
                    "DIFFICULTY: [junior/mid/senior]\n"
                    "SKILLS_TESTED: [list of skills]\n\n"
                    "Separate questions with ---\n\n"
                    "Generate questions now:"
                )
            )
            self._technical_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key="questions"
            )
        return self._technical_chain
    
    def _get_behavioral_chain(self) -> LLMChain:
        """Get or create behavioral question generation chain."""
        if self._behavioral_chain is None:
            prompt = PromptTemplate(
                input_variables=["job_description", "candidate_resume", "num_questions"],
                template=(
                    "You are an expert HR interviewer creating behavioral interview questions.\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "CANDIDATE RESUME:\n{candidate_resume}\n\n"
                    "Generate {num_questions} behavioral interview questions in STAR format that:\n"
                    "1. Test soft skills relevant to the role (leadership, teamwork, problem-solving, communication)\n"
                    "2. Are role-specific and relevant\n"
                    "3. Ask for STAR format responses (Situation, Task, Action, Result)\n"
                    "4. Include evaluation criteria\n\n"
                    "Format each question as:\n"
                    "QUESTION: [question text]\n"
                    "STAR_REQUIRED: [what to look for in STAR format]\n"
                    "SKILLS_ASSESSED: [soft skills being tested]\n"
                    "EVALUATION_CRITERIA: [what makes a good answer]\n\n"
                    "Separate questions with ---\n\n"
                    "Generate questions now:"
                )
            )
            self._behavioral_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key="questions"
            )
        return self._behavioral_chain
    
    def _get_followup_chain(self) -> LLMChain:
        """Get or create follow-up question generation chain."""
        if self._followup_chain is None:
            prompt = PromptTemplate(
                input_variables=["original_question", "candidate_answer", "job_description", "previous_answers"],
                template=(
                    "You are an expert interviewer analyzing a candidate's answer and generating follow-up questions.\n\n"
                    "ORIGINAL QUESTION:\n{original_question}\n\n"
                    "CANDIDATE ANSWER:\n{candidate_answer}\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "PREVIOUS ANSWERS CONTEXT:\n{previous_answers}\n\n"
                    "Analyze the answer and generate 1-2 follow-up questions that:\n"
                    "1. Probe deeper into interesting points mentioned\n"
                    "2. Clarify vague or incomplete answers\n"
                    "3. Test specific skills or experiences mentioned\n"
                    "4. Are relevant to the job requirements\n\n"
                    "Format as:\n"
                    "FOLLOWUP_1: [question text]\n"
                    "REASON: [why this follow-up is needed]\n"
                    "FOLLOWUP_2: [optional second question]\n"
                    "REASON: [why this follow-up is needed]\n\n"
                    "If no follow-up is needed, respond with: NO_FOLLOWUP: Answer is sufficient\n\n"
                    "Generate follow-up questions now:"
                )
            )
            self._followup_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key="followups"
            )
        return self._followup_chain
    
    def generate_technical_questions(
        self,
        job_description: str,
        candidate_resume: str,
        num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate technical interview questions.
        
        Args:
            job_description: Job description with required skills
            candidate_resume: Candidate's resume text
            num_questions: Number of questions to generate (default: 3)
        
        Returns:
            List of question dictionaries with:
            - question: Question text
            - expected_points: Key points to look for
            - difficulty: junior/mid/senior
            - skills_tested: List of skills
            - type: "technical"
        """
        try:
            chain = self._get_technical_chain()
            result = chain.invoke({
                "job_description": job_description,
                "candidate_resume": candidate_resume,
                "num_questions": num_questions
            })
            
            questions_text = result.get("questions", result.get("analysis", ""))
            questions = self._parse_questions(questions_text, "technical")
            
            logger.info(f"Generated {len(questions)} technical questions")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating technical questions: {e}")
            return []
    
    def generate_behavioral_questions(
        self,
        job_description: str,
        candidate_resume: str,
        num_questions: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate behavioral interview questions (STAR format).
        
        Args:
            job_description: Job description
            candidate_resume: Candidate's resume text
            num_questions: Number of questions to generate (default: 2)
        
        Returns:
            List of question dictionaries with:
            - question: Question text
            - star_required: What to look for in STAR format
            - skills_assessed: Soft skills being tested
            - evaluation_criteria: What makes a good answer
            - type: "behavioral"
        """
        try:
            chain = self._get_behavioral_chain()
            result = chain.invoke({
                "job_description": job_description,
                "candidate_resume": candidate_resume,
                "num_questions": num_questions
            })
            
            questions_text = result.get("questions", result.get("analysis", ""))
            questions = self._parse_behavioral_questions(questions_text)
            
            logger.info(f"Generated {len(questions)} behavioral questions")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating behavioral questions: {e}")
            return []
    
    def generate_followup_questions(
        self,
        original_question: str,
        candidate_answer: str,
        job_description: str,
        previous_answers: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate follow-up questions based on candidate's answer.
        
        Args:
            original_question: The original question asked
            candidate_answer: Candidate's answer
            job_description: Job description for context
            previous_answers: Context from previous answers (optional)
        
        Returns:
            List of follow-up question dictionaries, or empty list if none needed
        """
        try:
            chain = self._get_followup_chain()
            result = chain.invoke({
                "original_question": original_question,
                "candidate_answer": candidate_answer,
                "job_description": job_description,
                "previous_answers": previous_answers or "No previous answers."
            })
            
            followups_text = result.get("followups", result.get("analysis", ""))
            followups = self._parse_followup_questions(followups_text)
            
            logger.info(f"Generated {len(followups)} follow-up questions")
            return followups
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def _parse_questions(self, text: str, question_type: str) -> List[Dict[str, Any]]:
        """Parse technical questions from LLM output."""
        questions = []
        sections = text.split("---")
        
        for section in sections:
            if not section.strip():
                continue
            
            question_dict = {
                "type": question_type,
                "question": "",
                "expected_points": "",
                "difficulty": "mid",
                "skills_tested": []
            }
            
            # Extract question
            if "QUESTION:" in section:
                q_start = section.find("QUESTION:") + len("QUESTION:")
                q_end = section.find("EXPECTED_POINTS:", q_start)
                if q_end == -1:
                    q_end = section.find("DIFFICULTY:", q_start)
                question_dict["question"] = section[q_start:q_end].strip() if q_end > 0 else section[q_start:].strip()
            
            # Extract expected points
            if "EXPECTED_POINTS:" in section:
                ep_start = section.find("EXPECTED_POINTS:") + len("EXPECTED_POINTS:")
                ep_end = section.find("DIFFICULTY:", ep_start)
                if ep_end == -1:
                    ep_end = section.find("SKILLS_TESTED:", ep_start)
                question_dict["expected_points"] = section[ep_start:ep_end].strip() if ep_end > 0 else section[ep_start:].strip()
            
            # Extract difficulty
            if "DIFFICULTY:" in section:
                d_start = section.find("DIFFICULTY:") + len("DIFFICULTY:")
                d_end = section.find("SKILLS_TESTED:", d_start)
                difficulty = section[d_start:d_end].strip() if d_end > 0 else section[d_start:].strip()
                question_dict["difficulty"] = difficulty.lower() if difficulty else "mid"
            
            # Extract skills tested
            if "SKILLS_TESTED:" in section:
                st_start = section.find("SKILLS_TESTED:") + len("SKILLS_TESTED:")
                skills_text = section[st_start:].strip()
                # Parse comma-separated or newline-separated skills
                skills = [s.strip() for s in skills_text.replace("\n", ",").split(",") if s.strip()]
                question_dict["skills_tested"] = skills
            
            if question_dict["question"]:
                questions.append(question_dict)
        
        return questions
    
    def _parse_behavioral_questions(self, text: str) -> List[Dict[str, Any]]:
        """Parse behavioral questions from LLM output."""
        questions = []
        sections = text.split("---")
        
        for section in sections:
            if not section.strip():
                continue
            
            question_dict = {
                "type": "behavioral",
                "question": "",
                "star_required": "",
                "skills_assessed": [],
                "evaluation_criteria": ""
            }
            
            # Extract question
            if "QUESTION:" in section:
                q_start = section.find("QUESTION:") + len("QUESTION:")
                q_end = section.find("STAR_REQUIRED:", q_start)
                if q_end == -1:
                    q_end = section.find("SKILLS_ASSESSED:", q_start)
                question_dict["question"] = section[q_start:q_end].strip() if q_end > 0 else section[q_start:].strip()
            
            # Extract STAR requirements
            if "STAR_REQUIRED:" in section:
                sr_start = section.find("STAR_REQUIRED:") + len("STAR_REQUIRED:")
                sr_end = section.find("SKILLS_ASSESSED:", sr_start)
                if sr_end == -1:
                    sr_end = section.find("EVALUATION_CRITERIA:", sr_start)
                question_dict["star_required"] = section[sr_start:sr_end].strip() if sr_end > 0 else section[sr_start:].strip()
            
            # Extract skills assessed
            if "SKILLS_ASSESSED:" in section:
                sa_start = section.find("SKILLS_ASSESSED:") + len("SKILLS_ASSESSED:")
                sa_end = section.find("EVALUATION_CRITERIA:", sa_start)
                skills_text = section[sa_start:sa_end].strip() if sa_end > 0 else section[sa_start:].strip()
                skills = [s.strip() for s in skills_text.replace("\n", ",").split(",") if s.strip()]
                question_dict["skills_assessed"] = skills
            
            # Extract evaluation criteria
            if "EVALUATION_CRITERIA:" in section:
                ec_start = section.find("EVALUATION_CRITERIA:") + len("EVALUATION_CRITERIA:")
                question_dict["evaluation_criteria"] = section[ec_start:].strip()
            
            if question_dict["question"]:
                questions.append(question_dict)
        
        return questions
    
    def _parse_followup_questions(self, text: str) -> List[Dict[str, Any]]:
        """Parse follow-up questions from LLM output."""
        followups = []
        
        # Check if no follow-up needed
        if "NO_FOLLOWUP:" in text.upper():
            return []
        
        # Extract follow-up questions
        for i in range(1, 3):  # Check for FOLLOWUP_1 and FOLLOWUP_2
            followup_key = f"FOLLOWUP_{i}:"
            reason_key = f"REASON:"
            
            if followup_key in text:
                f_start = text.find(followup_key) + len(followup_key)
                f_end = text.find(reason_key, f_start)
                if f_end == -1:
                    f_end = text.find("FOLLOWUP_", f_start)
                question_text = text[f_start:f_end].strip() if f_end > 0 else text[f_start:].strip()
                
                # Extract reason
                reason_text = ""
                if reason_key in text[f_start:]:
                    r_start = text.find(reason_key, f_start) + len(reason_key)
                    r_end = text.find("FOLLOWUP_", r_start)
                    if r_end == -1:
                        r_end = len(text)
                    reason_text = text[r_start:r_end].strip()
                
                if question_text:
                    followups.append({
                        "question": question_text,
                        "reason": reason_text,
                        "type": "followup"
                    })
        
        return followups


# Singleton instance
_question_generator = None


def get_question_generator(llm=None) -> QuestionGenerator:
    """
    Get or create question generator instance (singleton).
    
    Args:
        llm: Optional LLM instance
    
    Returns:
        QuestionGenerator instance
    """
    global _question_generator
    if _question_generator is None:
        _question_generator = QuestionGenerator(llm=llm)
    return _question_generator

