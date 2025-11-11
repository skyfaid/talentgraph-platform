"""
CV Ranking system using hybrid RAG + LLM approach.

This module implements a hybrid ranking system that combines:
1. Semantic similarity (vector embeddings) - finds candidates with similar skills/experience
2. LLM-based evaluation - deep analysis of candidate-job fit

The final score is a weighted combination of both approaches, providing
both speed (from embeddings) and intelligence (from LLM analysis).

Architecture:
- Retrieval: Semantic search using ChromaDB vectorstore
- Augmentation: LLM evaluation of each candidate
- Ranking: Hybrid scoring (default: 30% semantic + 70% LLM)
"""
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import LLMChain
from typing import List, Dict, Optional
from ..utils.config import (
    SEMANTIC_WEIGHT,
    LLM_WEIGHT,
    DEFAULT_TOP_K,
    RANKING_SEARCH_MULTIPLIER,
    RESUME_HEAD_CHARS,
    RESUME_TAIL_CHARS,
    MIN_SCORE_THRESHOLD
)
from ..utils.candidate_generator import generate_candidate_info_deterministic
from ..utils.text_utils import prepare_resume_text, extract_score
from ..utils.logger import setup_logger

logger = setup_logger("cv_ranker")


class CVRanker:
    """
    CV Ranking system that combines semantic search with LLM analysis.
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        evaluation_chain: LLMChain,
        semantic_weight: float = None,
        llm_weight: float = None
    ):
        """
        Initialize CV Ranker with hybrid scoring approach.
        
        The ranker uses a two-stage approach:
        1. Fast semantic search to find relevant candidates
        2. LLM evaluation to deeply analyze each candidate
        
        Args:
            vectorstore: Chroma vectorstore containing resume embeddings.
                        Each document should have metadata with 'id', 'category', 'source'.
            evaluation_chain: LLMChain that evaluates resume-job fit.
                             Takes 'resume_text' and 'job_description' as inputs.
            semantic_weight: Weight for semantic similarity score (0-1).
                           Default: 0.3 (30% weight on embeddings)
            llm_weight: Weight for LLM evaluation score (0-1).
                       Default: 0.7 (70% weight on LLM analysis)
                       
        Note:
            Weights are automatically normalized if they don't sum to 1.0.
            Higher llm_weight = more emphasis on deep analysis.
            Higher semantic_weight = more emphasis on keyword/skill matching.
        """
        self.vectorstore = vectorstore
        self.evaluation_chain = evaluation_chain
        self.semantic_weight = semantic_weight or SEMANTIC_WEIGHT
        self.llm_weight = llm_weight or LLM_WEIGHT
        
        # Ensure weights sum to 1.0 (normalize if needed)
        total_weight = self.semantic_weight + self.llm_weight
        if abs(total_weight - 1.0) > 0.001:  # Allow small floating point differences
            logger.warning(
                f"Weights don't sum to 1.0 ({total_weight}). Normalizing..."
            )
            self.semantic_weight = self.semantic_weight / total_weight
            self.llm_weight = self.llm_weight / total_weight
        
        logger.info(
            f"Initialized CVRanker with weights: "
            f"semantic={self.semantic_weight:.2f}, llm={self.llm_weight:.2f}"
        )
    
    def rank_resumes(
        self,
        job_description: str,
        top_k: int = None,
        head_chars: int = None,
        tail_chars: int = None
    ) -> List[Dict]:
        """
        Rank resumes against a job description using hybrid scoring.
        
        Process:
        1. Semantic Search: Find top candidates using vector similarity
        2. LLM Evaluation: Deep analysis of each candidate's fit
        3. Hybrid Scoring: Combine both scores with weighted average
        4. Ranking: Sort by final score and return top K
        
        Args:
            job_description: Job description text to match against.
                           Should include required skills, experience, etc.
            top_k: Number of top candidates to return.
                  Default: 5 (from config)
            head_chars: Characters to keep from start of resume for LLM.
                      Resumes can be long, so we keep head + tail.
                      Default: 6000 (from config)
            tail_chars: Characters to keep from end of resume for LLM.
                       Default: 3000 (from config)
            
        Returns:
            List of ranked candidate dictionaries, each containing:
            - 'final_score': Combined hybrid score (0-10)
            - 'llm_score': LLM evaluation score (0-10)
            - 'semantic_similarity': Vector similarity score (0-1)
            - 'evaluation': Full LLM evaluation text
            - 'meta': Candidate metadata (id, category, source)
            - 'preview': First 240 chars of resume text
            
        Example:
            >>> ranker = CVRanker(vectorstore, evaluation_chain)
            >>> results = ranker.rank_resumes(
            ...     "Senior Data Engineer with Python, AWS, 5+ years",
            ...     top_k=10
            ... )
            >>> print(f"Top candidate score: {results[0]['final_score']}/10")
        """
        if top_k is None:
            top_k = DEFAULT_TOP_K
        if head_chars is None:
            head_chars = RESUME_HEAD_CHARS
        if tail_chars is None:
            tail_chars = RESUME_TAIL_CHARS
        
        logger.info(f"Ranking resumes for job description (top_k={top_k})")
        
        # Stage 1: Semantic search - retrieve more candidates than needed
        # This gives the LLM more options to evaluate
        search_k = max(2 * top_k, 12)
        logger.debug(f"Retrieving {search_k} candidates via semantic search")
        
        pairs = self.vectorstore.similarity_search_with_relevance_scores(
            job_description,
            k=search_k
        )
        
        logger.info(f"Found {len(pairs)} candidates, evaluating with LLM...")
        ranked = []
        
        # Stage 2: LLM evaluation for each candidate
        for idx, (doc, sim) in enumerate(pairs, 1):
            try:
                # Prepare resume text (keep head + tail to stay within token limits)
                prepared_text = prepare_resume_text(
                    doc.page_content,
                    head_chars=head_chars,
                    tail_chars=tail_chars
                )
                
                # Get LLM evaluation (this is the expensive operation)
                # Use invoke() instead of run() for newer LangChain versions
                evaluation_result = self.evaluation_chain.invoke({
                    "resume_text": prepared_text,
                    "job_description": job_description
                })
                
                # Extract content if result is a dict (LLMChain returns dict with output_key)
                if isinstance(evaluation_result, dict):
                    evaluation_result = evaluation_result.get("analysis", str(evaluation_result))
                
                # Extract numeric score from LLM's text evaluation
                llm_score = extract_score(evaluation_result)
                
                # Stage 3: Hybrid scoring
                # - Semantic similarity is 0-1, scale to 0-10
                # - LLM score is already 0-10
                # - Weighted combination
                final_score = round(
                    self.semantic_weight * (sim * 10) + self.llm_weight * llm_score,
                    2
                )
                
                # Ensure metadata has name and email (generate if missing)
                meta = doc.metadata.copy()
                candidate_id = meta.get("id", f"unknown_{idx}")
                
                # If name or email is missing, generate them deterministically
                if "name" not in meta or "email" not in meta:
                    candidate_info = generate_candidate_info_deterministic(candidate_id)
                    meta.setdefault("name", candidate_info["name"])
                    meta.setdefault("email", candidate_info["email"])
                
                ranked.append({
                    "evaluation": evaluation_result,
                    "meta": meta,
                    "preview": doc.page_content[:240].replace("\n", " "),
                    "semantic_similarity": round(sim, 3),
                    "llm_score": llm_score,
                    "final_score": final_score
                })
                
                logger.debug(
                    f"Candidate {idx}/{len(pairs)}: "
                    f"final={final_score:.2f}, llm={llm_score:.2f}, sim={sim:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error evaluating candidate {idx}: {e}")
                # Continue with other candidates even if one fails
                continue
        
        # Stage 4: Sort by final score (descending)
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Filter out candidates below minimum score threshold
        # This prevents showing bad matches even if top_k is high
        filtered_ranked = [r for r in ranked if r["final_score"] >= MIN_SCORE_THRESHOLD]
        
        # Return top K from filtered results (may be less than top_k if not enough good candidates)
        top_results = filtered_ranked[:top_k]
        
        if len(filtered_ranked) < len(ranked):
            logger.info(
                f"Filtered out {len(ranked) - len(filtered_ranked)} candidates below threshold ({MIN_SCORE_THRESHOLD}/10)"
            )
        
        if top_results:
            logger.info(
                f"Ranking complete. Top score: {top_results[0]['final_score']:.2f}/10 "
                f"(avg: {sum(r['final_score'] for r in top_results) / len(top_results):.2f}), "
                f"showing {len(top_results)}/{top_k} requested (filtered by min threshold: {MIN_SCORE_THRESHOLD})"
            )
        else:
            logger.warning(
                f"No candidates met the minimum score threshold ({MIN_SCORE_THRESHOLD}/10). "
                f"Consider adjusting the threshold or job description."
            )
        
        return top_results
    
    def get_retriever(self, k: int = 5):
        """
        Get a retriever for semantic search.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

