"""
Factory for creating embedding managers
Allows easy switching between different embedding approaches
"""

import os
from typing import Union
from .embedding_manager import EmbeddingManager
from .colbert_adapter import ColBERTEmbeddingManager

def create_embedding_manager(persist_directory: str = None) -> Union[EmbeddingManager, ColBERTEmbeddingManager]:
    """
    Create an embedding manager based on configuration
    
    Args:
        persist_directory: Directory to store embeddings
        
    Returns:
        EmbeddingManager or ColBERTEmbeddingManager instance
    """
    if persist_directory is None:
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./stored_embeddings")
    
    # Check which embedding approach to use
    embedding_type = os.getenv("EMBEDDING_TYPE", "colbert").lower()
    
    if embedding_type == "colbert":
        logger.info("Using ColBERT embedding manager")
        return ColBERTEmbeddingManager(persist_directory)
    elif embedding_type == "sentence_transformer":
        logger.info("Using SentenceTransformer embedding manager")
        return EmbeddingManager(persist_directory)
    else:
        logger.warning(f"Unknown embedding type: {embedding_type}, defaulting to ColBERT")
        return ColBERTEmbeddingManager(persist_directory)

# For backward compatibility
def get_embedding_manager():
    """Get embedding manager instance (for dependency injection)"""
    return create_embedding_manager()
