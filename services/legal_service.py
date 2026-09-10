"""
Legal Service — document indexing and management
"""
import logging
from typing import Dict, Any, Optional

from rag.chunking import chunk_text
from rag.embeddings import embed_texts
from rag.vector_store import get_vector_store
from services.document_parser import parse_document
from config import Config

logger = logging.getLogger(__name__)


def index_document(
    filepath: str,
    metadata: Dict[str, Any],
    store_path: str = None,
) -> int:
    """
    Parse, chunk, embed, and index a document.
    Returns the number of chunks indexed.
    """
    text = parse_document(filepath)
    if not text.strip():
        raise ValueError("No readable text could be extracted from the document.")

    chunks = chunk_text(
        text,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        metadata=metadata,
    )
    if not chunks:
        raise ValueError("Document produced no usable chunks.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=Config.EMBEDDING_MODEL)

    store = get_vector_store(store_path or Config.VECTOR_STORE_PATH)
    store.add_chunks(chunks, embeddings)

    logger.info(f"Indexed {len(chunks)} chunks from '{metadata.get('source', 'unknown')}'")
    return len(chunks)


def get_knowledge_base_stats() -> Dict[str, Any]:
    """Return stats about the current vector store."""
    store = get_vector_store()
    unique_sources = set()
    jurisdictions = {}
    doc_types = {}
    for chunk in store.chunks:
        meta = chunk.get("metadata", {})
        src = meta.get("source", "Unknown")
        unique_sources.add(src)
        jur = meta.get("jurisdiction", "India")
        jurisdictions[jur] = jurisdictions.get(jur, 0) + 1
        dt = meta.get("doc_type", "General")
        doc_types[dt] = doc_types.get(dt, 0) + 1

    return {
        "total_chunks": store.total_chunks,
        "unique_documents": len(unique_sources),
        "documents": list(unique_sources),
        "jurisdictions": jurisdictions,
        "doc_types": doc_types,
    }
