"""
Embedding generation for LawWise RAG.

Strategy (in order of preference):
1. Sentence Transformers (if available + torch installed)
2. Hash-based TF-IDF embeddings (numpy only, always available)

The hash-based fallback produces 512-dimensional dense vectors by:
  - Tokenising text into n-grams
  - Hashing each n-gram to a bucket (MurmurHash-like via Python hash())
  - Summing normalised TF-IDF weights per bucket
  - L2-normalising the result

This gives semantically useful similarity for same-domain legal text
without requiring any heavy ML dependencies.
"""
import re
import math
import hashlib
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

# ── Try to load Sentence Transformers ────────────────────────
_st_model = None
_st_available = False

def _try_load_st(model_name: str):
    global _st_model, _st_available
    if _st_available and _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading Sentence Transformers model: {model_name}")
        _st_model = SentenceTransformer(model_name)
        _st_available = True
        logger.info("Sentence Transformers model loaded.")
        return _st_model
    except Exception as e:
        logger.info(f"Sentence Transformers not available: {e}. Using TF-IDF fallback.")
        _st_available = False
        return None


# ── TF-IDF Hash Embedding ─────────────────────────────────────
EMBEDDING_DIM = 512  # dimensionality for hash embeddings
NGRAM_RANGE = (1, 2)  # unigrams + bigrams


def _tokenise(text: str) -> List[str]:
    """Lowercase word tokenisation."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z][a-z0-9]{1,}\b', text)
    return tokens


def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Produce a dense embedding via hash-based TF-IDF."""
    tokens = _tokenise(text)
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    # Build all n-grams
    all_ngrams = []
    for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
        all_ngrams.extend(_ngrams(tokens, n))

    if not all_ngrams:
        all_ngrams = tokens

    # TF: term frequency in this document
    tf: dict = {}
    for ng in all_ngrams:
        tf[ng] = tf.get(ng, 0) + 1
    total = sum(tf.values())

    # Build dense vector
    vec = np.zeros(dim, dtype=np.float32)
    for ng, count in tf.items():
        # Deterministic bucket via SHA-256
        h = int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16)
        bucket = h % dim
        weight = (1 + math.log(count / total + 1e-9))
        vec[bucket] += weight

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Public API ────────────────────────────────────────────────

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Return the ST model if available, else None (hash fallback)."""
    return _try_load_st(model_name)


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = _try_load_st(model_name)
    if model is not None:
        try:
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.warning(f"ST encoding failed: {e}. Falling back to TF-IDF.")

    # TF-IDF hash fallback
    logger.info(f"Generating hash-TF-IDF embeddings for {len(texts)} texts.")
    return np.stack([_hash_embedding(t) for t in texts], axis=0).astype(np.float32)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embedding for a single query."""
    return embed_texts([query], model_name)
