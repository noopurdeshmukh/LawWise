"""
RAG Retriever — semantic search over the LawWise vector store
"""
import os
import logging
from typing import List, Dict, Any, Optional

from rag.embeddings import embed_query
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = None,
    filters: Optional[Dict[str, Any]] = None,
    store_path: str = None,
    max_per_source: int = 2,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks for a query.

    - similarity_threshold: read from SIMILARITY_THRESHOLD env var (default 0.05).
      Hash-TF-IDF embeddings produce lower cosine scores than neural embeddings.
    - max_per_source: cap chunks per unique source document so one large document
      cannot fill all slots and crowd out other sources.

    Returns:
        List of dicts: {text, metadata, score}
    """
    if similarity_threshold is None:
        similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", "0.05"))

    store = get_vector_store(store_path)
    if store.total_chunks == 0:
        logger.warning("Vector store is empty. No documents indexed yet.")
        return []

    query_emb = embed_query(query)
    # Fetch more candidates so diversity filtering still yields top_k results
    raw = store.search(
        query_emb,
        top_k=top_k * 8,
        similarity_threshold=similarity_threshold,
        filters=filters,
    )

    # Apply per-source diversity cap
    source_counts: Dict[str, int] = {}
    results = []
    for chunk in raw:
        src = chunk.get("metadata", {}).get("source", "unknown")
        if source_counts.get(src, 0) >= max_per_source:
            continue
        source_counts[src] = source_counts.get(src, 0) + 1
        results.append(chunk)
        if len(results) >= top_k:
            break

    logger.info(
        f"Retrieved {len(results)} diverse chunks (from {len(source_counts)} sources) "
        f"for query: '{query[:60]}'"
    )
    return results


def format_context_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        section = meta.get("section", "")
        page = meta.get("page", "")
        source_str = f"[Source {i}: {source}"
        if section:
            source_str += f", {section}"
        if page:
            source_str += f", Page {page}"
        source_str += "]"
        parts.append(f"{source_str}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def get_source_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build citation objects from retrieved chunks."""
    seen = set()
    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        if source in seen:
            continue
        seen.add(source)
        citations.append({
            "document": source,
            "section": meta.get("section", ""),
            "page": meta.get("page", ""),
            "source_type": meta.get("doc_type", "Legal Document"),
            "jurisdiction": meta.get("jurisdiction", "India"),
            "year": meta.get("year", ""),
            "relevance_score": round(chunk.get("score", 0), 3),
            "excerpt": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
        })
    return citations
