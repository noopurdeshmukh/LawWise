import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from rag.vector_store import get_vector_store
store = get_vector_store()
print("Total chunks:", store.total_chunks)
print("Index file exists:", os.path.exists(os.path.join(store.store_path, "lawwise_index.pkl")))
print("Embeddings file exists:", os.path.exists(os.path.join(store.store_path, "lawwise_embeddings.npy")))
print("Store path:", store.store_path)

if store.total_chunks > 0:
    from rag.retriever import retrieve
    results = retrieve("indemnity contract", top_k=3, similarity_threshold=0.05)
    print(f"Retrieval results: {len(results)}")
    for r in results:
        print(f"  score={r['score']:.3f} src={r['metadata'].get('source','?')[:50]}")
