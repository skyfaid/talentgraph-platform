"""
Vectorstore management for resume embeddings.
"""
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import List, Optional
from pathlib import Path
from ..utils.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from .embedder import create_embeddings


def create_vectorstore(
    documents: List[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_directory: Optional[Path] = None,
    collection_name: str = None
) -> Chroma:
    """
    Create or load a Chroma vectorstore with resume documents.
    
    Args:
        documents: List of LangChain Document objects
        embeddings: Embeddings model. If None, creates default.
        persist_directory: Directory to persist vectorstore. If None, uses config default.
        collection_name: Name of the collection. If None, uses config default.
        
    Returns:
        Chroma vectorstore instance
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR
    
    if collection_name is None:
        collection_name = CHROMA_COLLECTION_NAME
    
    print(f"🗄️ Creating/Loading Chroma vectorstore at {persist_directory}...")
    
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name
    )
    
    # Persist to disk
    try:
        vectorstore.persist()
    except Exception:
        # Some langchain/chroma versions persist automatically
        pass
    
    print(f"✅ Stored {len(documents)} resumes in Chroma (collection='{collection_name}').")
    return vectorstore


def load_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_directory: Optional[Path] = None,
    collection_name: str = None
) -> Chroma:
    """
    Load an existing Chroma vectorstore.
    
    Args:
        embeddings: Embeddings model. If None, creates default.
        persist_directory: Directory where vectorstore is persisted. If None, uses config default.
        collection_name: Name of the collection. If None, uses config default.
        
    Returns:
        Chroma vectorstore instance
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR
    
    if collection_name is None:
        collection_name = CHROMA_COLLECTION_NAME
    
    print(f"📂 Loading Chroma vectorstore from {persist_directory}...")
    
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    return vectorstore


def create_documents_from_resumes(resumes: List[dict]) -> List[Document]:
    """
    Create LangChain Document objects from resume dictionaries.
    
    Args:
        resumes: List of resume dictionaries with 'text', 'id', 'category', 'source', 'name', 'email' keys
        
    Returns:
        List of Document objects
    """
    documents = [
        Document(
            page_content=r["text"],
            metadata={
                "id": r["id"],
                "category": r.get("category", "Unknown"),
                "source": r.get("source", "Unknown"),
                "name": r.get("name", "Unknown Candidate"),
                "email": r.get("email", "unknown@example.com")
            }
        )
        for r in resumes
    ]
    return documents

