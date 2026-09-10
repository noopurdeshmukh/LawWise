"""
Text chunking for RAG pipeline
"""
import re
from typing import List, Dict, Any


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    metadata: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata.
    Returns list of dicts with 'text' and 'metadata'.
    """
    metadata = metadata or {}
    sentences = _split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk).strip()
            if chunk_text_str:
                chunk_meta = {**metadata, "chunk_index": len(chunks)}
                chunks.append({"text": chunk_text_str, "metadata": chunk_meta})
            # Overlap: keep last portion
            overlap_chars = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                overlap_chars += len(s)
                overlap_sentences.insert(0, s)
                if overlap_chars >= chunk_overlap:
                    break
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunk_text_str = " ".join(current_chunk).strip()
        if chunk_text_str:
            chunk_meta = {**metadata, "chunk_index": len(chunks)}
            chunks.append({"text": chunk_text_str, "metadata": chunk_meta})

    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Further split very long parts at paragraph boundaries
    sentences = []
    for part in parts:
        if len(part) > 500:
            sub_parts = part.split("\n\n")
            sentences.extend([s.strip() for s in sub_parts if s.strip()])
        else:
            if part.strip():
                sentences.append(part.strip())
    return sentences
