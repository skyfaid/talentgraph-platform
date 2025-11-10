"""
Metrics calculation and evaluation utilities for CV ranking system.
Provides metrics for model evaluation and monitoring.
"""
from typing import List, Dict, Optional, Any
import numpy as np
from ..utils.logger import setup_logger

logger = setup_logger("metrics")


def calculate_ranking_metrics(
    ranked_results: List[Dict],
    ground_truth: Optional[List[str]] = None,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Calculate metrics for ranking performance.
    
    Args:
        ranked_results: List of ranked candidate dictionaries with 'final_score', 'llm_score', etc.
        ground_truth: Optional list of ground truth candidate IDs (for evaluation)
        top_k: Number of top candidates to consider
        
    Returns:
        Dictionary of metrics:
        - avg_final_score: Average final score
        - avg_llm_score: Average LLM score
        - avg_semantic_similarity: Average semantic similarity
        - score_std: Standard deviation of final scores
        - top_k_avg_score: Average score of top K candidates
        
    Example:
        >>> results = ranker.rank_resumes("Data Engineer", top_k=10)
        >>> metrics = calculate_ranking_metrics(results, top_k=5)
        >>> print(f"Average score: {metrics['avg_final_score']:.2f}")
    """
    if not ranked_results:
        logger.warning("No results provided for metrics calculation")
        return {}
    
    # Extract scores
    final_scores = [r.get('final_score', 0.0) for r in ranked_results]
    llm_scores = [r.get('llm_score', 0.0) for r in ranked_results]
    semantic_sims = [r.get('semantic_similarity', 0.0) for r in ranked_results]
    
    metrics = {
        'avg_final_score': float(np.mean(final_scores)),
        'avg_llm_score': float(np.mean(llm_scores)),
        'avg_semantic_similarity': float(np.mean(semantic_sims)),
        'score_std': float(np.std(final_scores)),
        'min_score': float(np.min(final_scores)),
        'max_score': float(np.max(final_scores)),
    }
    
    # Top K metrics
    top_k_results = ranked_results[:top_k]
    if top_k_results:
        top_k_scores = [r.get('final_score', 0.0) for r in top_k_results]
        metrics['top_k_avg_score'] = float(np.mean(top_k_scores))
        metrics['top_k_min_score'] = float(np.min(top_k_scores))
        metrics['top_k_max_score'] = float(np.max(top_k_scores))
    
    # Score distribution percentiles
    metrics['score_p25'] = float(np.percentile(final_scores, 25))
    metrics['score_p50'] = float(np.percentile(final_scores, 50))  # median
    metrics['score_p75'] = float(np.percentile(final_scores, 75))
    
    logger.debug(f"Calculated {len(metrics)} ranking metrics")
    return metrics


def calculate_category_consistency(
    ranked_results: List[Dict],
    query_category: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate category consistency metrics (how many results match query category).
    
    Args:
        ranked_results: List of ranked results with metadata containing 'category'
        query_category: Optional expected category for the query
        
    Returns:
        Dictionary with consistency metrics
    """
    if not ranked_results:
        return {}
    
    categories = [r.get('meta', {}).get('category', 'Unknown') for r in ranked_results]
    
    metrics = {
        'unique_categories': len(set(categories)),
        'total_candidates': len(categories)
    }
    
    if query_category:
        matching = sum(1 for cat in categories if cat == query_category)
        metrics['category_match_rate'] = matching / len(categories) if categories else 0.0
        metrics['category_matches'] = matching
    
    # Most common category
    from collections import Counter
    category_counts = Counter(categories)
    if category_counts:
        most_common = category_counts.most_common(1)[0]
        metrics['most_common_category'] = most_common[0]
        metrics['most_common_count'] = most_common[1]
        metrics['most_common_percentage'] = most_common[1] / len(categories)
    
    return metrics


def log_evaluation_metrics(
    ranked_results: List[Dict],
    job_description: str,
    tracker: Optional[Any] = None
) -> Dict[str, float]:
    """
    Calculate and log evaluation metrics to MLflow (if tracker provided).
    
    Args:
        ranked_results: List of ranked candidate results
        job_description: Job description used for ranking
        tracker: Optional MLflowTracker instance for logging
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate all metrics
    ranking_metrics = calculate_ranking_metrics(ranked_results)
    category_metrics = calculate_category_consistency(ranked_results)
    
    # Combine metrics
    all_metrics = {**ranking_metrics, **category_metrics}
    
    # Log to MLflow if tracker provided
    if tracker:
        try:
            tracker.log_metrics(all_metrics)
            logger.info(f"Logged {len(all_metrics)} metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    return all_metrics

