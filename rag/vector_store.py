"""
Vector store for LawWise RAG.
Uses FAISS if available, falls back to a pure-numpy cosine similarity store.
"""
import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS backend available.")
except ImportError:
    logger.info("FAISS not available. Using numpy cosine similarity fallback.")

INDEX_FILE = "lawwise_index.pkl"
EMBEDDINGS_FILE = "lawwise_embeddings.npy"


class VectorStore:
    """
    Hybrid vector store: FAISS when available, numpy cosine similarity otherwise.
    Both backends expose the same public interface.
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.chunks: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self._faiss_index = None
        os.makedirs(store_path, exist_ok=True)
        self._load()

    # ── Persistence ─────────────────────────────────────────

    def _index_path(self):
        return os.path.join(self.store_path, INDEX_FILE)

    def _emb_path(self):
        return os.path.join(self.store_path, EMBEDDINGS_FILE)

    def _faiss_path(self):
        return os.path.join(self.store_path, "lawwise.faiss")

    def _load(self):
        try:
            if os.path.exists(self._index_path()):
                with open(self._index_path(), "rb") as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks from store.")

            if FAISS_AVAILABLE and os.path.exists(self._faiss_path()):
                self._faiss_index = faiss.read_index(self._faiss_path())
                logger.info("FAISS index loaded.")
            elif os.path.exists(self._emb_path()):
                self._embeddings = np.load(self._emb_path())
                logger.info(f"Numpy embeddings loaded: shape {self._embeddings.shape}")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            self.chunks = []
            self._embeddings = None
            self._faiss_index = None

    def save(self):
        try:
            with open(self._index_path(), "wb") as f:
                pickle.dump(self.chunks, f)
            if FAISS_AVAILABLE and self._faiss_index is not None:
                faiss.write_index(self._faiss_index, self._faiss_path())
            elif self._embeddings is not None:
                np.save(self._emb_path(), self._embeddings)
            logger.info(f"Vector store saved: {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    # ── Indexing ─────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add new chunks + their embeddings to the store."""
        embeddings = embeddings.astype(np.float32)

        if FAISS_AVAILABLE:
            if self._faiss_index is None:
                dim = embeddings.shape[1]
                self._faiss_index = faiss.IndexFlatIP(dim)  # inner-product (cosine after norm)
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            norm_emb = embeddings / norms
            self._faiss_index.add(norm_emb)
        else:
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

        self.chunks.extend(chunks)
        self.save()
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar chunks above the similarity threshold."""
        if len(self.chunks) == 0:
            return []

        query_embedding = query_embedding.astype(np.float32)

        if FAISS_AVAILABLE and self._faiss_index is not None:
            return self._search_faiss(query_embedding, top_k, similarity_threshold, filters)
        elif self._embeddings is not None:
            return self._search_numpy(query_embedding, top_k, similarity_threshold, filters)
        return []

    def _search_faiss(self, query_emb, top_k, threshold, filters):
        # Normalize query
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        k = min(top_k * 4, len(self.chunks))
        scores, indices = self._faiss_index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim = float(score)  # cosine similarity (0–1)
            if sim < threshold:
                continue
            chunk = self.chunks[idx]
            if not self._passes_filter(chunk, filters):
                continue
            results.append({
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "score": sim,
            })
            if len(results) >= top_k:
                break
        return results

    def _search_numpy(self, query_emb, top_k, threshold, filters):
        # Cosine similarity
        q = query_emb.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        emb = self._embeddings  # (N, D)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        norm_emb = emb / norms  # (N, D)

        similarities = norm_emb.dot(q)  # (N,)
        # Get top candidates
        k = min(top_k * 4, len(similarities))
        top_indices = np.argsort(-similarities)[:k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < threshold:
                break
            chunk = self.chunks[idx]
            if not self._passes_filter(chunk, filters):
                continue
            results.append({
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "score": sim,
            })
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _passes_filter(chunk, filters):
        if not filters:
            return True
        meta = chunk.get("metadata", {})
        for k, v in filters.items():
            if v and str(meta.get(k, "")).lower() != str(v).lower():
                return False
        return True

    # ── Deletion ─────────────────────────────────────────────

    def delete_by_document(self, doc_name: str):
        """Remove all chunks for a document and rebuild the index."""
        new_chunks = [
            c for c in self.chunks
            if c.get("metadata", {}).get("source") != doc_name
        ]
        if len(new_chunks) == len(self.chunks):
            return

        if not new_chunks:
            self.chunks = []
            self._embeddings = None
            self._faiss_index = None
            self.save()
            return

        from rag.embeddings import embed_texts
        texts = [c["text"] for c in new_chunks]
        embeddings = embed_texts(texts)
        self.chunks = []
        self._embeddings = None
        self._faiss_index = None
        self.add_chunks(new_chunks, embeddings)
        logger.info(f"Rebuilt index after deleting '{doc_name}'. Chunks: {len(self.chunks)}")

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


# ── Module-level singleton ────────────────────────────────────
_store: Optional[VectorStore] = None


def get_vector_store(store_path: str = None) -> VectorStore:
    global _store
    if _store is None:
        if store_path is None:
            from config import Config
            store_path = Config.VECTOR_STORE_PATH
        _store = VectorStore(store_path)
    return _store
