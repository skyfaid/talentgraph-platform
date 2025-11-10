"""
FastAPI application for CV Ranking System.

Endpoints:
- POST /rank - Rank CVs against a job description
- POST /upload - Upload PDF CVs
- GET /health - Health check
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

from src.api import (
    RankRequest,
    RankResponse,
    CandidateResult,
    CandidateResultWithExplanation,
    HealthResponse,
    UploadResponse,
    Explanation,
    ExplainRequest,
    SkillAnalysis,
    ScoreBreakdown,
    SHAPAnalysis,
    LIMEAnalysis,
    get_service
)
from src.xai import RankingExplainer
from src.pdf import extract_text_from_multiple_pdfs
from src.utils.logger import setup_logger
from pathlib import Path
import tempfile
import os

logger = setup_logger("fastapi_app")

# Create FastAPI app
app = FastAPI(
    title="CV Ranking API",
    description="AI-powered CV ranking system using hybrid RAG + LLM approach",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("🚀 Starting CV Ranking API...")
    service = get_service()
    # Check if vectorstore exists and has names
    # If not, recreate it to include names from CSV
    success = service.initialize(use_existing_vectorstore=True)
    if not success:
        logger.error("⚠️ Service initialization failed - some endpoints may not work")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CV Ranking API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the service and its components.
    """
    service = get_service()
    
    return HealthResponse(
        status="healthy" if service.is_ready() else "not ready",
        vectorstore_ready=service.vectorstore is not None,
        llm_ready=service.llm is not None,
        total_resumes=service.get_total_resumes()
    )


@app.post("/rank", response_model=RankResponse, tags=["Ranking"])
async def rank_candidates(request: RankRequest):
    """
    Rank CVs against a job description.
    
    This endpoint:
    1. Performs semantic search to find relevant candidates
    2. Evaluates each candidate with LLM
    3. Combines scores (30% semantic + 70% LLM)
    4. Returns top K ranked candidates
    5. Optionally includes XAI explanations
    
    Args:
        request: RankRequest with job_description, top_k, and include_explanations
        
    Returns:
        RankResponse with ranked candidates (with explanations if requested)
    """
    service = get_service()
    
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Please check /health endpoint."
        )
    
    try:
        logger.info(
            f"Ranking request: job_description length={len(request.job_description)}, "
            f"top_k={request.top_k}, include_explanations={request.include_explanations}"
        )
        
        # Rank candidates
        results = service.ranker.rank_resumes(
            job_description=request.job_description,
            top_k=request.top_k
        )
        
        explainer = RankingExplainer() if request.include_explanations else None
        
        # Convert to response model
        candidates = []
        for r in results:
            candidate = CandidateResult(
                final_score=r['final_score'],
                llm_score=r['llm_score'],
                semantic_similarity=r['semantic_similarity'],
                evaluation=r['evaluation'],
                meta=r['meta'],
                preview=r['preview']
            )
            
            # Add explanation if requested
            if request.include_explanations and explainer:
                try:
                    # Get full resume text from vectorstore
                    # Search for the document by metadata
                    doc_results = service.vectorstore.similarity_search_with_relevance_scores(
                        r['preview'], k=1
                    )
                    resume_text = doc_results[0][0].page_content if doc_results else r['preview']
                    
                    logger.debug(f"Generating explanation for candidate {r['meta'].get('id')}")
                    
                    # Generate explanation (with SHAP and LIME if available)
                    explanation_dict = explainer.generate_explanation(
                        candidate_result=r,
                        job_description=request.job_description,
                        resume_text=resume_text,
                        vectorstore=service.vectorstore,
                        ranker=service.ranker
                    )
                    
                    # Convert to Explanation model
                    score_breakdown_data = explanation_dict['score_breakdown']
                    skill_analysis_data = explanation_dict['skill_analysis']
                    
                    # Build SHAP and LIME analysis if available
                    shap_analysis = None
                    if 'shap_analysis' in explanation_dict:
                        shap_data = explanation_dict['shap_analysis']
                        shap_analysis = SHAPAnalysis(
                            available=shap_data.get('available', False),
                            skill_importance=shap_data.get('skill_importance'),
                            error=shap_data.get('error')
                        )
                    
                    lime_analysis = None
                    if 'lime_analysis' in explanation_dict:
                        lime_data = explanation_dict['lime_analysis']
                        lime_analysis = LIMEAnalysis(
                            available=lime_data.get('available', False),
                            text_importance=lime_data.get('text_importance'),
                            error=lime_data.get('error')
                        )
                    
                    explanation = Explanation(
                        candidate_id=explanation_dict['candidate_id'],
                        overall_score=explanation_dict['overall_score'],
                        score_breakdown=ScoreBreakdown(
                            final_score=score_breakdown_data['final_score'],
                            semantic_contribution=score_breakdown_data['components']['semantic']['contribution'],
                            llm_contribution=score_breakdown_data['components']['llm']['contribution'],
                            semantic_score=score_breakdown_data['components']['semantic']['normalized_score'],
                            llm_score=score_breakdown_data['components']['llm']['raw_score'],
                            formula=score_breakdown_data['formula']
                        ),
                        skill_analysis=SkillAnalysis(**skill_analysis_data),
                        llm_evaluation=explanation_dict['llm_evaluation'],
                        strengths=explanation_dict['insights']['strengths'],
                        weaknesses=explanation_dict['insights']['weaknesses'],
                        experience_info=explanation_dict['insights']['experience'],
                        leadership_info=explanation_dict['insights']['leadership'],
                        primary_factor=explanation_dict['ranking_factors']['primary'],
                        secondary_factors=explanation_dict['ranking_factors']['secondary'],
                        shap_analysis=shap_analysis,
                        lime_analysis=lime_analysis
                    )
                    
                    # Create candidate with explanation - include explanation in dict
                    candidate_dict = candidate.dict()
                    # Serialize explanation properly
                    try:
                        # Try Pydantic v2 method first
                        if hasattr(explanation, 'model_dump'):
                            explanation_dict_serialized = explanation.model_dump()
                        else:
                            # Fall back to Pydantic v1
                            explanation_dict_serialized = explanation.dict()
                    except Exception as ser_error:
                        logger.warning(f"Error serializing explanation: {ser_error}")
                        # Last resort: convert to dict manually
                        explanation_dict_serialized = {
                            'candidate_id': explanation.candidate_id,
                            'overall_score': explanation.overall_score,
                            'score_breakdown': explanation.score_breakdown.dict() if hasattr(explanation.score_breakdown, 'dict') else explanation.score_breakdown.model_dump() if hasattr(explanation.score_breakdown, 'model_dump') else explanation.score_breakdown,
                            'skill_analysis': explanation.skill_analysis.dict() if hasattr(explanation.skill_analysis, 'dict') else explanation.skill_analysis.model_dump() if hasattr(explanation.skill_analysis, 'model_dump') else explanation.skill_analysis,
                            'llm_evaluation': explanation.llm_evaluation,
                            'strengths': explanation.strengths,
                            'weaknesses': explanation.weaknesses,
                            'experience_info': explanation.experience_info,
                            'leadership_info': explanation.leadership_info,
                            'primary_factor': explanation.primary_factor,
                            'secondary_factors': explanation.secondary_factors,
                            'shap_analysis': explanation.shap_analysis.dict() if explanation.shap_analysis and hasattr(explanation.shap_analysis, 'dict') else (explanation.shap_analysis.model_dump() if explanation.shap_analysis and hasattr(explanation.shap_analysis, 'model_dump') else explanation.shap_analysis),
                            'lime_analysis': explanation.lime_analysis.dict() if explanation.lime_analysis and hasattr(explanation.lime_analysis, 'dict') else (explanation.lime_analysis.model_dump() if explanation.lime_analysis and hasattr(explanation.lime_analysis, 'model_dump') else explanation.lime_analysis)
                        }
                    
                    candidate_dict['explanation'] = explanation_dict_serialized
                    candidate_with_explanation = CandidateResultWithExplanation(**candidate_dict)
                    candidates.append(candidate_with_explanation)
                    logger.info(f"✅ Explanation added for candidate {r['meta'].get('id')}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate explanation for candidate {r['meta'].get('id')}: {e}", exc_info=True)
                    candidates.append(candidate)
            else:
                candidates.append(candidate)
        
        logger.info(f"✅ Ranking complete: {len(candidates)} candidates returned")
        
        # Serialize candidates properly - ensure explanations are included
        serialized_candidates = []
        for cand in candidates:
            try:
                # Check if it's a Pydantic model
                if hasattr(cand, 'model_dump'):
                    cand_dict = cand.model_dump()
                elif hasattr(cand, 'dict'):
                    cand_dict = cand.dict()
                else:
                    # Already a dict
                    cand_dict = cand
                
                # Ensure explanation is included if present
                if hasattr(cand, 'explanation') and cand.explanation is not None:
                    if hasattr(cand.explanation, 'model_dump'):
                        cand_dict['explanation'] = cand.explanation.model_dump()
                    elif hasattr(cand.explanation, 'dict'):
                        cand_dict['explanation'] = cand.explanation.dict()
                    else:
                        cand_dict['explanation'] = cand.explanation
                
                serialized_candidates.append(cand_dict)
            except Exception as ser_err:
                logger.error(f"Error serializing candidate: {ser_err}")
                # Fallback: basic serialization
                if isinstance(cand, dict):
                    serialized_candidates.append(cand)
                else:
                    serialized_candidates.append({
                        'final_score': getattr(cand, 'final_score', 0),
                        'llm_score': getattr(cand, 'llm_score', 0),
                        'semantic_similarity': getattr(cand, 'semantic_similarity', 0),
                        'evaluation': getattr(cand, 'evaluation', ''),
                        'meta': getattr(cand, 'meta', {}),
                        'preview': getattr(cand, 'preview', ''),
                        'explanation': getattr(cand, 'explanation', None)
                    })
        
        return RankResponse(
            job_description=request.job_description,
            top_k=request.top_k,
            candidates=serialized_candidates,
            total_candidates_evaluated=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error ranking candidates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ranking candidates: {str(e)}"
        )


@app.post("/explain", response_model=Explanation, tags=["XAI"])
async def explain_candidate(request: ExplainRequest):
    """
    Get detailed explanation for why a specific candidate was ranked.
    
    This endpoint provides XAI (Explainable AI) insights:
    - Score breakdown (semantic vs LLM contribution)
    - Skill matching analysis
    - Strengths and weaknesses
    - Experience and leadership assessment
    - Primary and secondary ranking factors
    
    Args:
        request: ExplainRequest with candidate_id and job_description
        
    Returns:
        Explanation with detailed analysis
    """
    service = get_service()
    
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Please check /health endpoint."
        )
    
    try:
        # Find candidate in vectorstore
        # Search by candidate ID in metadata
        # Note: This is a simplified approach - in production you'd have a better lookup
        search_results = service.vectorstore.similarity_search_with_relevance_scores(
            request.candidate_id, k=10
        )
        
        candidate_doc = None
        for doc, _ in search_results:
            if doc.metadata.get('id') == request.candidate_id:
                candidate_doc = doc
                break
        
        if not candidate_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Candidate with ID '{request.candidate_id}' not found"
            )
        
        # Rank this specific candidate to get full result
        results = service.ranker.rank_resumes(
            job_description=request.job_description,
            top_k=50  # Get more results to find our candidate
        )
        
        candidate_result = None
        for r in results:
            if r['meta'].get('id') == request.candidate_id:
                candidate_result = r
                break
        
        if not candidate_result:
            # If not in top results, create a basic result
            # Re-evaluate this specific candidate
            from ..utils.text_utils import prepare_resume_text, extract_score
            prepared_text = prepare_resume_text(candidate_doc.page_content)
            evaluation_result = service.evaluation_chain.invoke({
                "resume_text": prepared_text,
                "job_description": request.job_description
            })
            if isinstance(evaluation_result, dict):
                evaluation_result = evaluation_result.get("analysis", str(evaluation_result))
            
            llm_score = extract_score(evaluation_result)
            
            # Get similarity
            similarity_results = service.vectorstore.similarity_search_with_relevance_scores(
                request.job_description, k=100
            )
            similarity = 0.0
            for doc, sim in similarity_results:
                if doc.metadata.get('id') == request.candidate_id:
                    similarity = sim
                    break
            
            final_score = 0.3 * (similarity * 10) + 0.7 * llm_score
            
            candidate_result = {
                'final_score': round(final_score, 2),
                'llm_score': llm_score,
                'semantic_similarity': round(similarity, 3),
                'evaluation': evaluation_result,
                'meta': candidate_doc.metadata,
                'preview': candidate_doc.page_content[:240].replace("\n", " ")
            }
        
        # Generate explanation (with SHAP and LIME if available)
        explainer = RankingExplainer()
        explanation_dict = explainer.generate_explanation(
            candidate_result=candidate_result,
            job_description=request.job_description,
            resume_text=candidate_doc.page_content,
            vectorstore=service.vectorstore,
            ranker=service.ranker
        )
        
        # Convert to Explanation model
        score_breakdown_data = explanation_dict['score_breakdown']
        skill_analysis_data = explanation_dict['skill_analysis']
        
        # Build SHAP and LIME analysis if available
        shap_analysis = None
        if 'shap_analysis' in explanation_dict:
            shap_data = explanation_dict['shap_analysis']
            shap_analysis = SHAPAnalysis(
                available=shap_data.get('available', False),
                skill_importance=shap_data.get('skill_importance'),
                error=shap_data.get('error')
            )
        
        lime_analysis = None
        if 'lime_analysis' in explanation_dict:
            lime_data = explanation_dict['lime_analysis']
            lime_analysis = LIMEAnalysis(
                available=lime_data.get('available', False),
                text_importance=lime_data.get('text_importance'),
                error=lime_data.get('error')
            )
        
        explanation = Explanation(
            candidate_id=explanation_dict['candidate_id'],
            overall_score=explanation_dict['overall_score'],
            score_breakdown=ScoreBreakdown(
                final_score=score_breakdown_data['final_score'],
                semantic_contribution=score_breakdown_data['components']['semantic']['contribution'],
                llm_contribution=score_breakdown_data['components']['llm']['contribution'],
                semantic_score=score_breakdown_data['components']['semantic']['normalized_score'],
                llm_score=score_breakdown_data['components']['llm']['raw_score'],
                formula=score_breakdown_data['formula']
            ),
            skill_analysis=SkillAnalysis(**skill_analysis_data),
            llm_evaluation=explanation_dict['llm_evaluation'],
            strengths=explanation_dict['insights']['strengths'],
            weaknesses=explanation_dict['insights']['weaknesses'],
            experience_info=explanation_dict['insights']['experience'],
            leadership_info=explanation_dict['insights']['leadership'],
            primary_factor=explanation_dict['ranking_factors']['primary'],
            secondary_factors=explanation_dict['ranking_factors']['secondary'],
            shap_analysis=shap_analysis,
            lime_analysis=lime_analysis
        )
        
        logger.info(f"✅ Explanation generated for candidate {request.candidate_id}")
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating explanation: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_pdf_cvs(files: List[UploadFile] = File(...)):
    """
    Upload PDF CVs to the system.
    
    This endpoint:
    1. Accepts multiple PDF files
    2. Extracts text from each PDF
    3. Adds them to the vectorstore (when implemented)
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        UploadResponse with processing status
    """
    service = get_service()
    
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized. Please check /health endpoint."
        )
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Save files temporarily and process
    processed = 0
    failed = 0
    temp_files = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files
            for file in files:
                if not file.filename.endswith('.pdf'):
                    failed += 1
                    logger.warning(f"Skipping non-PDF file: {file.filename}")
                    continue
                
                file_path = temp_path / file.filename
                with open(file_path, 'wb') as f:
                    content = await file.read()
                    f.write(content)
                temp_files.append(file_path)
            
            # Extract text from PDFs
            pdf_results = extract_text_from_multiple_pdfs(temp_files)
            
            # Process results
            pdf_texts = []
            for result in pdf_results:
                if 'error' in result:
                    failed += 1
                    logger.error(f"Error processing {result['filename']}: {result['error']}")
                else:
                    processed += 1
                    pdf_texts.append({
                        'text': result['text'],
                        'id': f"pdf_{processed}",
                        'category': 'Unknown',
                        'source': result['filename']
                    })
            
            # TODO: Add PDF resumes to vectorstore
            # For now, this is a placeholder
            logger.info(f"Processed {processed} PDFs, {failed} failed")
            
            return UploadResponse(
                message=f"Processed {processed} PDF files successfully",
                files_processed=processed,
                files_failed=failed,
                total_resumes=service.get_total_resumes() or 0
            )
            
    except Exception as e:
        logger.error(f"Error uploading PDFs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDFs: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

