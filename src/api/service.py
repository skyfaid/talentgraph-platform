"""
Service layer for initializing and managing CV ranking components.
"""
import os
from pathlib import Path
from typing import Optional

from ..data import (
    load_all_datasets,
    clean_resume_dataframe,
    combine_resume_datasets,
    resumes_to_dataframe
)
from ..embeddings import (
    create_embeddings,
    create_vectorstore,
    load_vectorstore,
    create_documents_from_resumes
)
from ..llm import initialize_llm, create_evaluation_chain
from ..ranker import CVRanker
from ..utils.config import CHROMA_DB_DIR
from ..utils.logger import setup_logger

logger = setup_logger("api_service")


class CVRankingService:
    """
    Service class that manages the CV ranking system components.
    Handles initialization, loading, and provides access to ranker.
    """
    
    def __init__(self):
        """Initialize the service (components loaded lazily)."""
        self.vectorstore = None
        self.ranker = None
        self.llm = None
        self.evaluation_chain = None
        self._initialized = False
        self._total_resumes = 0
    
    def initialize(
        self,
        force_reload: bool = False,
        use_existing_vectorstore: bool = True
    ) -> bool:
        """
        Initialize the CV ranking service.
        
        Args:
            force_reload: If True, reload data even if vectorstore exists
            use_existing_vectorstore: If True, use existing vectorstore if available
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing CV Ranking Service...")
            
            # Check if vectorstore exists
            vectorstore_exists = (CHROMA_DB_DIR / "chroma_db").exists() or CHROMA_DB_DIR.exists()
            
            if use_existing_vectorstore and vectorstore_exists and not force_reload:
                logger.info("Loading existing vectorstore...")
                embeddings = create_embeddings()
                self.vectorstore = load_vectorstore(embeddings)
                
                # Check if vectorstore has names in metadata (sample check)
                try:
                    sample_docs = self.vectorstore.similarity_search("test", k=1)
                    if sample_docs and 'name' not in sample_docs[0].metadata:
                        logger.warning("⚠️ Vectorstore doesn't have names. Recreating with names from CSV...")
                        vectorstore_exists = False  # Force recreation
                    else:
                        logger.info("✅ Vectorstore loaded from disk (with names)")
                except:
                    logger.warning("⚠️ Could not verify vectorstore metadata. Recreating...")
                    vectorstore_exists = False
            
            if not vectorstore_exists or force_reload:
                logger.info("Creating new vectorstore from datasets...")
                # Load and process data
                df1, df2 = load_all_datasets()
                df1_clean = clean_resume_dataframe(df1, text_column="Resume_str", category_column="Category")
                df2_clean = clean_resume_dataframe(df2, text_column="Resume", category_column="Category")
                
                all_resumes = combine_resume_datasets(
                    df1_clean, df2_clean,
                    id_column1="ID",
                    category_column="Category",
                    text_column="clean_text"
                )
                
                # Create vectorstore
                documents = create_documents_from_resumes(all_resumes)
                embeddings = create_embeddings()
                self.vectorstore = create_vectorstore(documents, embeddings)
                self._total_resumes = len(documents)
                logger.info(f"✅ Created vectorstore with {self._total_resumes} resumes")
            
            # Initialize LLM
            logger.info("Initializing LLM...")
            self.llm = initialize_llm()
            self.evaluation_chain = create_evaluation_chain(self.llm)
            
            # Create ranker
            logger.info("Creating CV Ranker...")
            self.ranker = CVRanker(self.vectorstore, self.evaluation_chain)
            
            self._initialized = True
            logger.info("✅ CV Ranking Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service: {e}")
            self._initialized = False
            return False
    
    def is_ready(self) -> bool:
        """Check if service is ready to use."""
        return self._initialized and self.ranker is not None
    
    def get_total_resumes(self) -> Optional[int]:
        """Get total number of resumes in vectorstore."""
        return self._total_resumes
    
    def add_pdf_resumes(self, pdf_texts: list[dict]) -> bool:
        """
        Add resumes from PDF texts to the vectorstore.
        
        Args:
            pdf_texts: List of dicts with 'text', 'id', 'category', 'source' keys
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore:
                logger.error("Vectorstore not initialized")
                return False
            
            # Create documents from PDF texts
            documents = create_documents_from_resumes(pdf_texts)
            
            # Add to existing vectorstore
            # Note: ChromaDB supports adding documents to existing collection
            embeddings = create_embeddings()
            new_vectorstore = create_vectorstore(documents, embeddings)
            
            # For now, we'll need to merge or recreate
            # This is a limitation - ChromaDB doesn't easily merge collections
            # In production, you'd want to add documents incrementally
            logger.warning("PDF resume addition requires vectorstore recreation - feature in progress")
            return False
            
        except Exception as e:
            logger.error(f"Failed to add PDF resumes: {e}")
            return False


# Global service instance
_service_instance: Optional[CVRankingService] = None


def get_service() -> CVRankingService:
    """Get or create the global service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CVRankingService()
    return _service_instance

