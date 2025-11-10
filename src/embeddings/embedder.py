"""
Embedding model initialization and management.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..utils.config import EMBEDDING_MODEL_NAME


def create_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings model.
    
    Args:
        model_name: Name of the embedding model. If None, uses default from config.
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME
    
    print(f"🔬 Initializing HuggingFaceEmbeddings ({model_name})...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

