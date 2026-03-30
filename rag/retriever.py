"""
Retriever — orchestrates query embedding + vector search + optional metadata filtering.
"""

from rag.embeddings import embed_query
from rag.vector_store import VectorStore


def retrieve(query: str, vector_store: VectorStore, k: int = 5,
             sender_filter: str = None, keyword_filter: str = None) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User's question.
        vector_store: The VectorStore instance.
        k: Number of results to retrieve.
        sender_filter: Optional — filter results by sender (case-insensitive substring).
        keyword_filter: Optional — filter results containing this keyword.

    Returns:
        List of relevant chunk dicts.
    """
    query_emb = embed_query(query)

    # Retrieve more than k if we're filtering, to compensate for filtered-out results
    fetch_k = k * 3 if (sender_filter or keyword_filter) else k
    results = vector_store.search(query_emb, k=fetch_k)

    # Apply metadata filters
    if sender_filter:
        sender_lower = sender_filter.lower()
        results = [r for r in results if sender_lower in r.get("metadata", {}).get("sender", "").lower()]

    if keyword_filter:
        kw_lower = keyword_filter.lower()
        results = [r for r in results if kw_lower in r.get("text", "").lower()]

    return results[:k]


def build_context(results: list[dict]) -> str:
    """
    Build a context string from retrieved chunks for the LLM.
    """
    if not results:
        return "No relevant email content found."

    context_parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        context_parts.append(
            f"--- Email Chunk {i} ---\n"
            f"Subject: {meta.get('subject', 'N/A')}\n"
            f"From: {meta.get('sender', 'N/A')}\n"
            f"Date: {meta.get('date', 'N/A')}\n"
            f"Content:\n{r['text']}\n"
        )

    return "\n".join(context_parts)
