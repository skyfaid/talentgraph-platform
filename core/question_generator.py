"""
Question Generation Module using Mistral 7B
FIXED: Job-specific questions with proper context injection
"""
import requests
import json
from typing import List, Dict, Optional

from utils.config import MISTRAL_MODEL


class QuestionGenerator:
    def __init__(self):
        """Initialize Mistral connection via Ollama"""
        self.model = MISTRAL_MODEL
        self.ollama_url = "http://localhost:11434/api/generate"
        self.conversation_history = []
        self.conversation_summary = ""
        
        # Topics to cover in the interview
        self.topics_covered = {
            "background": False,
            "motivation": False,
            "availability": False,
            "experience": False,
            "technical_skills": False
        }
    
    def generate_question(
        self, 
        previous_answer: Optional[str] = None,
        job_context: Optional[Dict] = None
    ) -> str:
        """
        Generate next interview question based on context
        
        Args:
            previous_answer: Candidate's previous answer
            job_context: Job details (title, skills, etc.)
            
        Returns:
            Generated question
        """
        # Build prompt based on what's been covered
        uncovered = [k for k, v in self.topics_covered.items() if not v]
        
        if not previous_answer:
            # First question
            prompt = self._get_initial_prompt(job_context)
        else:
            # Update conversation summary
            self._update_conversation_summary(previous_answer)
            # Follow-up question
            prompt = self._get_followup_prompt(previous_answer, uncovered, job_context)
        
        # Call Mistral
        response = self._call_mistral(prompt)
        
        # Update topics covered
        self._update_topics(response)
        
        return response
    
    def _get_initial_prompt(self, job_context: Optional[Dict] = None) -> str:
        """Get prompt for first question with job-specific context"""
        
        # Extract job details
        job_title = "a position"
        if job_context:
            job_title = job_context.get('title', 'a position')
        
        return f"""You are a professional recruiter conducting a phone screening interview for a {job_title} role.

Generate ONE concise, natural opening question to start the interview. The question should be relevant to {job_title}.

GOOD EXAMPLES for {job_title}:
- "Could you start by telling me about your background and what interests you in this {job_title} opportunity?"
- "I'd love to hear about your recent experience. What brings you to apply for this {job_title} role?"
- "Tell me about yourself and why you're interested in working as a {job_title}."

BAD EXAMPLES (avoid these):
- Generic "Tell me about yourself" without context
- Questions about unrelated fields
- Yes/no questions

Generate a natural, professional opening question under 25 words that is SPECIFIC to {job_title}.
Only output the question, nothing else."""
    
    def _update_conversation_summary(self, answer: str):
        """Build running summary of conversation"""
        if len(self.conversation_history) < 3:
            self.conversation_history.append(answer)
        else:
            # Keep only last 3 answers in memory
            self.conversation_history = self.conversation_history[-2:] + [answer]
        
        # Create brief summary
        key_points = []
        for ans in self.conversation_history:
            words = ans.split()[:15]  # First 15 words
            key_points.append(" ".join(words))
        self.conversation_summary = " | ".join(key_points)
    
    def _get_followup_prompt(
        self, 
        previous_answer: str, 
        uncovered_topics: List[str],
        job_context: Optional[Dict]
    ) -> str:
        """Generate prompt for follow-up questions with job-specific context"""
        
        # Extract job details
        job_title = "the position"
        required_skills = []
        
        if job_context:
            job_title = job_context.get('title', 'the position')
            required_skills = job_context.get('skills', [])
        
        # Build job context string
        job_info = f"Job Role: {job_title}\n"
        if required_skills:
            job_info += f"Required Skills: {', '.join(required_skills)}\n"
        
        topics_str = ", ".join(uncovered_topics) if uncovered_topics else "general fit"
        
        conversation_context = ""
        if self.conversation_summary:
            conversation_context = f"\nConversation so far: {self.conversation_summary}\n"
        
        # Add job-specific examples
        skill_examples = ""
        if required_skills and len(required_skills) > 0:
            skill_examples = f"\n- Ask about their experience with: {', '.join(required_skills[:3])}"
        
        prompt = f"""You are a professional recruiter conducting a phone screening interview for a {job_title} position.

{job_info}{conversation_context}
Candidate's last answer: "{previous_answer}"

Remaining topics to cover: {topics_str}

CRITICAL: Your question MUST be relevant to {job_title}. DO NOT ask about unrelated fields.

TASK: Generate ONE natural follow-up question that is SPECIFIC to {job_title}.

JOB-SPECIFIC QUESTION EXAMPLES for {job_title}:{skill_examples}
- "You mentioned [specific thing]. How did that experience relate to {job_title}?"
- "What aspects of {job_title} are you most excited about?"
- "Can you tell me about a relevant project or experience?"

RULES:
- Reference something specific from their answer when possible
- Keep under 30 words
- Be conversational and natural
- MUST be relevant to {job_title} - do not drift to other domains
- Ask open-ended questions

Generate the question now (only the question, no explanation):"""
        
        return prompt
    
    def _call_mistral(self, prompt: str) -> str:
        """
        Call Mistral via Ollama API with better parameters
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 100,
                    "num_predict": 100  # Limit length
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "").strip()
                
                # Clean up any extra formatting
                generated = generated.replace("Question:", "").strip()
                generated = generated.replace('"', '').strip()
                
                return generated
            else:
                print(f"Mistral API error: {response.status_code}")
                return self._get_fallback_question()
                
        except Exception as e:
            print(f"Error calling Mistral: {e}")
            return self._get_fallback_question()
    
    def _get_fallback_question(self) -> str:
        """Return fallback question if Mistral fails"""
        fallback_questions = [
            "Can you tell me about your background and recent experience?",
            "What interests you about this position?",
            "When would you be available to start?",
            "What are your salary expectations?",
            "What are you looking for in your next role?"
        ]
        
        # Return first uncovered topic question
        for topic, covered in self.topics_covered.items():
            if not covered:
                idx = list(self.topics_covered.keys()).index(topic)
                if idx < len(fallback_questions):
                    return fallback_questions[idx]
        
        return fallback_questions[0]
    
    def _update_topics(self, question: str):
        """Update which topics have been covered"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["background", "experience", "worked", "previous"]):
            self.topics_covered["background"] = True
            self.topics_covered["experience"] = True
        
        if any(word in question_lower for word in ["why", "interest", "motivate", "looking for"]):
            self.topics_covered["motivation"] = True
        
        if any(word in question_lower for word in ["start", "available", "notice", "when"]):
            self.topics_covered["availability"] = True
        
        if any(word in question_lower for word in ["skills", "technical", "technologies", "tools"]):
            self.topics_covered["technical_skills"] = True
    
    def get_coverage_status(self) -> Dict[str, bool]:
        """Return which topics have been covered"""
        return self.topics_covered.copy()