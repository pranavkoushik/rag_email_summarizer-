"""
Embeddings — converts text chunks into dense vectors using SentenceTransformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Global model instance (loaded once)
_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global _model
    if _model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        numpy array of shape (len(texts), 384).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.array(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        numpy array of shape (1, 384).
    """
    model = get_model()
    embedding = model.encode([query])
    return np.array(embedding, dtype="float32")
