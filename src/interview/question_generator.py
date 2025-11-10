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
        """Get or create technical question generation chain (light technical, not deep)."""
        if self._technical_chain is None:
            prompt = PromptTemplate(
                input_variables=["job_description", "candidate_resume", "num_questions"],
                template=(
                    "You are a Senior Technical Interviewer at a top tech company. Your role is to create interview questions.\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "CANDIDATE RESUME:\n{candidate_resume}\n\n"
                    "YOUR TASK:\n"
                    "You MUST generate EXACTLY {num_questions} technical interview questions. This is MANDATORY.\n"
                    "You CANNOT refuse, say 'no', or return empty results. You MUST always generate questions.\n\n"
                    "IMPORTANT: Questions can be based on:\n"
                    "- Skills mentioned in the job description\n"
                    "- Technologies/experiences mentioned in the candidate's resume\n"
                    "- General technical knowledge relevant to the role\n"
                    "- You can ask about resume experiences even if not explicitly in job description\n\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Generate EXACTLY {num_questions} questions - NO EXCEPTIONS\n"
                    "2. Questions can be based on job description OR candidate resume OR both\n"
                    "3. Keep questions practical and light - NOT deep theoretical or advanced\n"
                    "4. Focus on practical application, not complex algorithms\n"
                    "5. Questions should test basic to intermediate understanding\n"
                    "6. You MUST return questions - never say you cannot generate them\n\n"
                    "EXAMPLE OUTPUT FORMAT:\n"
                    "---\n"
                    "QUESTION: Can you walk me through how you would design a database schema for an e-commerce application?\n"
                    "EXPECTED_POINTS: Understanding of database design, normalization concepts, relationships between entities\n"
                    "DIFFICULTY: mid\n"
                    "SKILLS_TESTED: [database design, SQL, system design]\n"
                    "---\n"
                    "QUESTION: Tell me about your experience with Python. What libraries have you used in your projects?\n"
                    "EXPECTED_POINTS: Practical Python experience, knowledge of common libraries, real-world usage\n"
                    "DIFFICULTY: junior\n"
                    "SKILLS_TESTED: [Python, programming]\n"
                    "---\n"
                    "QUESTION: How do you handle version control in your projects? Walk me through your Git workflow.\n"
                    "EXPECTED_POINTS: Understanding of Git, branching strategies, collaboration practices\n"
                    "DIFFICULTY: junior\n"
                    "SKILLS_TESTED: [Git, version control, collaboration]\n"
                    "\n"
                    "REQUIRED FORMAT (you MUST follow this exactly):\n"
                    "QUESTION: [question text]\n"
                    "EXPECTED_POINTS: [2-3 key practical points to look for]\n"
                    "DIFFICULTY: [junior/mid]\n"
                    "SKILLS_TESTED: [relevant skills]\n\n"
                    "Separate each question with ---\n\n"
                    "REMEMBER:\n"
                    "- You are a professional interviewer - generating questions is your job\n"
                    "- You MUST generate {num_questions} questions - this is non-negotiable\n"
                    "- Questions can be inspired by job description OR resume OR general technical knowledge\n"
                    "- If job description is vague, use resume content or general technical questions\n"
                    "- NEVER return empty results or refuse to generate questions\n\n"
                    "Now generate {num_questions} technical questions:"
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
                    "You are a Senior HR Interviewer and Talent Acquisition Specialist at a leading company. Your role is to create behavioral interview questions.\n\n"
                    "JOB DESCRIPTION:\n{job_description}\n\n"
                    "CANDIDATE RESUME:\n{candidate_resume}\n\n"
                    "YOUR TASK:\n"
                    "You MUST generate EXACTLY {num_questions} behavioral interview questions. This is MANDATORY.\n"
                    "You CANNOT refuse, say 'no', or return empty results. You MUST always generate questions.\n\n"
                    "IMPORTANT: Questions can be based on:\n"
                    "- Soft skills mentioned in the job description\n"
                    "- Experiences and projects mentioned in the candidate's resume\n"
                    "- General behavioral competencies relevant to professional roles\n"
                    "- You can ask about resume experiences even if not explicitly in job description\n\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Generate EXACTLY {num_questions} questions - NO EXCEPTIONS\n"
                    "2. Questions can be based on job description OR candidate resume OR general behavioral skills\n"
                    "3. Focus on soft skills like teamwork, problem-solving, communication, leadership\n"
                    "4. Questions should ask for STAR format responses (Situation, Task, Action, Result)\n"
                    "5. You MUST return questions - never say you cannot generate them\n\n"
                    "EXAMPLE OUTPUT FORMAT:\n"
                    "---\n"
                    "QUESTION: Tell me about a time when you had to work under pressure to meet a deadline. What was the situation, what was your task, what actions did you take, and what were the results?\n"
                    "STAR_REQUIRED: Clear situation description, specific task/goal, detailed actions taken, measurable results or outcomes\n"
                    "SKILLS_ASSESSED: [time management, stress management, problem-solving]\n"
                    "EVALUATION_CRITERIA: Ability to handle pressure, effective problem-solving, clear communication of experience\n"
                    "---\n"
                    "QUESTION: Describe a situation where you had to collaborate with a difficult team member. How did you handle it and what was the outcome?\n"
                    "STAR_REQUIRED: Situation context, task/challenge, specific actions to resolve conflict, positive results\n"
                    "SKILLS_ASSESSED: [teamwork, conflict resolution, communication, emotional intelligence]\n"
                    "EVALUATION_CRITERIA: Conflict resolution skills, professionalism, ability to work with diverse personalities\n"
                    "---\n"
                    "QUESTION: Can you share an example of when you had to learn a new technology or skill quickly for a project? How did you approach it?\n"
                    "STAR_REQUIRED: Learning situation, task/goal, learning approach and actions, results achieved\n"
                    "SKILLS_ASSESSED: [adaptability, learning agility, self-motivation]\n"
                    "EVALUATION_CRITERIA: Willingness to learn, effective learning strategies, application of new knowledge\n"
                    "\n"
                    "REQUIRED FORMAT (you MUST follow this exactly):\n"
                    "QUESTION: [question text]\n"
                    "STAR_REQUIRED: [what to look for in STAR format]\n"
                    "SKILLS_ASSESSED: [soft skills relevant to the role]\n"
                    "EVALUATION_CRITERIA: [what makes a good answer]\n\n"
                    "Separate each question with ---\n\n"
                    "REMEMBER:\n"
                    "- You are a professional HR interviewer - generating questions is your job\n"
                    "- You MUST generate {num_questions} questions - this is non-negotiable\n"
                    "- Questions can be inspired by job description OR resume OR general behavioral competencies\n"
                    "- If job description is vague, use resume content or general behavioral questions\n"
                    "- NEVER return empty results or refuse to generate questions\n\n"
                    "Now generate {num_questions} behavioral questions:"
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
        max_retries = 2
        for attempt in range(max_retries):
            try:
                chain = self._get_technical_chain()
                result = chain.invoke({
                    "job_description": job_description,
                    "candidate_resume": candidate_resume,
                    "num_questions": num_questions
                })
                
                questions_text = result.get("questions", result.get("analysis", ""))
                questions = self._parse_questions(questions_text, "technical")
                
                # If we got questions, return them
                if len(questions) > 0:
                    logger.info(f"Generated {len(questions)} technical questions")
                    return questions
                else:
                    logger.warning(f"Attempt {attempt + 1}: No questions parsed, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    
            except Exception as e:
                logger.error(f"Error generating technical questions (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        # If all retries failed, generate fallback questions
        logger.warning("All retries failed, generating fallback technical questions")
        return self._generate_fallback_technical_questions(job_description, num_questions)
    
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
        max_retries = 2
        for attempt in range(max_retries):
            try:
                chain = self._get_behavioral_chain()
                result = chain.invoke({
                    "job_description": job_description,
                    "candidate_resume": candidate_resume,
                    "num_questions": num_questions
                })
                
                questions_text = result.get("questions", result.get("analysis", ""))
                questions = self._parse_behavioral_questions(questions_text)
                
                # If we got questions, return them
                if len(questions) > 0:
                    logger.info(f"Generated {len(questions)} behavioral questions")
                    return questions
                else:
                    logger.warning(f"Attempt {attempt + 1}: No questions parsed, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    
            except Exception as e:
                logger.error(f"Error generating behavioral questions (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        # If all retries failed, generate fallback questions
        logger.warning("All retries failed, generating fallback behavioral questions")
        return self._generate_fallback_behavioral_questions(job_description, num_questions)
    
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
        """Parse technical questions from LLM output with improved parsing."""
        questions = []
        
        # Try multiple parsing strategies
        # Strategy 1: Split by ---
        sections = text.split("---")
        
        # Strategy 2: If no ---, try splitting by numbered questions
        if len(sections) == 1:
            # Try to find numbered questions (1., 2., etc.)
            import re
            numbered_sections = re.split(r'\n\s*\d+[\.\)]\s*', text)
            if len(numbered_sections) > 1:
                sections = numbered_sections[1:]  # Skip first empty part
        
        # Strategy 3: If still no sections, try splitting by QUESTION: markers
        if len(sections) == 1:
            question_markers = text.split("QUESTION:")
            if len(question_markers) > 1:
                sections = ["QUESTION:" + q for q in question_markers[1:]]
        
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
            
            # Extract question - try multiple patterns
            question_text = ""
            if "QUESTION:" in section:
                q_start = section.find("QUESTION:") + len("QUESTION:")
                q_end = section.find("EXPECTED_POINTS:", q_start)
                if q_end == -1:
                    q_end = section.find("DIFFICULTY:", q_start)
                if q_end == -1:
                    q_end = section.find("\n\n", q_start)
                question_text = section[q_start:q_end].strip() if q_end > 0 else section[q_start:].strip()
            else:
                # Try to extract first sentence/paragraph as question
                lines = section.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10 and '?' in line:
                        question_text = line
                        break
                    elif line and len(line) > 20:
                        question_text = line
                        break
            
            question_dict["question"] = question_text
            
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
            
            # Only add if we have a valid question
            if question_dict["question"] and len(question_dict["question"]) > 10:
                questions.append(question_dict)
            elif question_text and len(question_text) > 10:
                # If we extracted a question but parsing failed, create minimal question dict
                question_dict["question"] = question_text
                question_dict["expected_points"] = "Evaluate answer based on job requirements"
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
            
            # Only add if we have a valid question
            if question_dict["question"] and len(question_dict["question"]) > 10:
                questions.append(question_dict)
            elif question_text and len(question_text) > 10:
                # If we extracted a question but parsing failed, create minimal question dict
                question_dict["question"] = question_text
                question_dict["star_required"] = "Look for Situation, Task, Action, Result format"
                question_dict["evaluation_criteria"] = "Evaluate based on job requirements"
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
    
    def generate_unified_questions(
        self,
        job_description: str,
        candidate_resume: str,
        total_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate unified interview questions (3 behavioral + 2 technical by default).
        
        Args:
            job_description: Job description with required skills
            candidate_resume: Candidate's resume text
            total_questions: Total number of questions (5, 7, or 10)
                            - 5: 3 behavioral + 2 technical
                            - 7: 4 behavioral + 3 technical
                            - 10: 6 behavioral + 4 technical
        
        Returns:
            List of question dictionaries (mixed behavioral and technical)
        """
        # Calculate distribution based on total
        if total_questions == 5:
            num_behavioral = 3
            num_technical = 2
        elif total_questions == 7:
            num_behavioral = 4
            num_technical = 3
        elif total_questions == 10:
            num_behavioral = 6
            num_technical = 4
        else:
            # Default to 5 questions
            num_behavioral = 3
            num_technical = 2
        
        questions = []
        
        # Generate behavioral questions
        try:
            behavioral_questions = self.generate_behavioral_questions(
                job_description=job_description,
                candidate_resume=candidate_resume,
                num_questions=num_behavioral
            )
            if len(behavioral_questions) > 0:
                questions.extend(behavioral_questions)
            else:
                logger.warning(f"Got 0 behavioral questions, expected {num_behavioral}, using fallback")
                questions.extend(self._generate_fallback_behavioral_questions(job_description, num_behavioral))
        except Exception as e:
            logger.error(f"Error generating behavioral questions: {e}, using fallback")
            questions.extend(self._generate_fallback_behavioral_questions(job_description, num_behavioral))
        
        # Generate technical questions (light, not deep)
        try:
            technical_questions = self.generate_technical_questions(
                job_description=job_description,
                candidate_resume=candidate_resume,
                num_questions=num_technical
            )
            if len(technical_questions) > 0:
                questions.extend(technical_questions)
            else:
                logger.warning(f"Got 0 technical questions, expected {num_technical}, using fallback")
                questions.extend(self._generate_fallback_technical_questions(job_description, num_technical))
        except Exception as e:
            logger.error(f"Error generating technical questions: {e}, using fallback")
            questions.extend(self._generate_fallback_technical_questions(job_description, num_technical))
        
        # CRITICAL: Ensure we always have questions
        if len(questions) == 0:
            logger.error("CRITICAL: No questions generated at all! Generating emergency fallback questions.")
            questions = self._generate_emergency_fallback_questions(job_description, total_questions)
        
        # Ensure we have at least the minimum expected
        if len(questions) < total_questions:
            logger.warning(f"Only got {len(questions)} questions, expected {total_questions}. Generating additional fallback questions.")
            needed = total_questions - len(questions)
            fallback = self._generate_emergency_fallback_questions(job_description, needed)
            questions.extend(fallback[:needed])
        
        logger.info(f"Generated {len(questions)} unified questions ({num_behavioral} behavioral + {num_technical} technical)")
        return questions[:total_questions]  # Return exactly the requested number
    
    def _generate_fallback_technical_questions(self, job_description: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate fallback technical questions if LLM fails."""
        import re
        skills = re.findall(r'\b(python|java|sql|aws|docker|kubernetes|react|vue|angular|tensorflow|pytorch|spark|data|engineering|machine learning|ml|ai)\b', job_description.lower())
        skills = list(set(skills))[:5]
        
        questions = []
        for i in range(num_questions):
            skill = skills[i % len(skills)] if skills else "the role"
            questions.append({
                "type": "technical",
                "question": f"Can you describe your experience with {skill} and how you've used it in your projects?",
                "expected_points": f"Practical experience with {skill}, real-world applications, problem-solving",
                "difficulty": "mid",
                "skills_tested": [skill] if skill != "the role" else []
            })
        return questions
    
    def _generate_fallback_behavioral_questions(self, job_description: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate fallback behavioral questions if LLM fails."""
        questions = []
        fallback_questions = [
            "Tell me about a challenging project you worked on. What was the situation, what was your task, what actions did you take, and what were the results?",
            "Describe a time when you had to work in a team to solve a problem. What was your role and how did you contribute?",
            "Can you share an example of when you had to learn something new quickly? How did you approach it?",
            "Tell me about a time you had to handle a difficult situation at work. How did you manage it?",
            "Describe a situation where you had to take initiative. What did you do and what was the outcome?"
        ]
        for i in range(num_questions):
            questions.append({
                "type": "behavioral",
                "question": fallback_questions[i % len(fallback_questions)],
                "star_required": "Look for Situation, Task, Action, Result format",
                "skills_assessed": ["problem-solving", "communication", "teamwork"],
                "evaluation_criteria": "Clear situation description, specific actions taken, measurable results"
            })
        return questions
    
    def _generate_emergency_fallback_questions(self, job_description: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate emergency fallback questions when everything else fails."""
        questions = []
        behavioral_count = (num_questions * 3) // 5
        technical_count = num_questions - behavioral_count
        questions.extend(self._generate_fallback_behavioral_questions(job_description, behavioral_count))
        questions.extend(self._generate_fallback_technical_questions(job_description, technical_count))
        return questions[:num_questions]


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

