"""
Groq Cloud LLM service for resume analysis and ranking.

This module handles integration with Groq Cloud's fast LLM inference.
Groq provides optimized inference for models like Llama, making it ideal
for real-time CV ranking where we need to evaluate many candidates.

Key Features:
- Fast inference (optimized for production)
- Configurable model parameters (temperature, top_p, etc.)
- Reproducible results (seed support)
- Error handling and logging
"""
import os
from langchain_groq import ChatGroq

# LangChain imports - use langchain_classic for chains
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from ..utils.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TEMPERATURE,
    GROQ_TOP_P,
    GROQ_MAX_TOKENS,
    GROQ_SEED
)
from ..utils.logger import setup_logger

logger = setup_logger("groq_service")


def initialize_llm(
    api_key: str = None,
    model_name: str = None,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    seed: int = None
) -> ChatGroq:
    """
    Initialize Groq Cloud LLM.
    
    Args:
        api_key: Groq API key. If None, uses environment variable or config.
        model_name: Model name. If None, uses config default.
        temperature: Temperature setting. If None, uses config default.
        top_p: Top-p setting. If None, uses config default.
        max_tokens: Max tokens. If None, uses config default.
        seed: Random seed. If None, uses config default.
        
    Returns:
        ChatGroq LLM instance
    """
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    
    if not api_key or api_key == "":
        raise ValueError("GROQ_API_KEY not found. Please set it in environment or config.")
    
    if model_name is None:
        model_name = GROQ_MODEL_NAME
    if temperature is None:
        temperature = GROQ_TEMPERATURE
    if top_p is None:
        top_p = GROQ_TOP_P
    if max_tokens is None:
        max_tokens = GROQ_MAX_TOKENS
    if seed is None:
        seed = GROQ_SEED
    
    try:
        # top_p and seed should be in model_kwargs for newer versions
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={
                "top_p": top_p,
                "seed": seed  # Seed ensures reproducibility for experiments
            }
        )
        logger.info(
            f"✅ Groq Cloud LLM initialized: {model_name} "
            f"(temp={temperature}, top_p={top_p}, max_tokens={max_tokens})"
        )
        return llm
    except Exception as e:
        logger.error(f"❌ Groq initialization failed: {e}")
        raise


def create_evaluation_chain(llm: ChatGroq) -> LLMChain:
    """
    Create LLM chain for resume evaluation - exactly as in notebook.
    
    Args:
        llm: ChatGroq LLM instance
        
    Returns:
        LLMChain for evaluation
    """
    augmentation_prompt = PromptTemplate(
        input_variables=["resume_text", "job_description"],
        template=(
            "You are an HR expert reviewing a candidate's resume against a job description.\n"
            "Read the JOB DESCRIPTION and RESUME carefully.\n"
            "Provide a concise, human-friendly evaluation with:\n"
            "1) A score out of 10 reflecting overall suitability.\n"
            "2) Explanation of matched skills, missing skills, experience, leadership, and tech level.\n"
            "3) Reasons why this candidate is a strong or weak match.\n\n"
            "JOB DESCRIPTION:\n{job_description}\n\n"
            "RESUME:\n{resume_text}\n\n"
            "Your output should be human-readable text, like:\n"
            "\"Score: 7/10. The candidate matches Python and AWS skills, lacks Airflow experience. "
            "They have 4 years experience, slightly below requirement of 5 years. Leadership experience is present. Overall fit is good because...\""
        )
    )
    
    augmentation_chain = LLMChain(llm=llm, prompt=augmentation_prompt, output_key="analysis")
    
    return augmentation_chain

