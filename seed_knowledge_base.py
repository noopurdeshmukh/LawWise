"""
Script to index sample documents into the LawWise knowledge base.
Run: python seed_knowledge_base.py
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from config import Config
from rag.ingestion import extract_text_from_file, clean_text
from rag.chunking import chunk_text
from rag.embeddings import embed_texts
from rag.vector_store import get_vector_store

SAMPLE_DOCS = [
    {
        "filename": "sample_indian_contract_act.txt",
        "source": "Indian Contract Act, 1872 — Key Provisions",
        "doc_type": "Act / Statute",
        "jurisdiction": "India",
        "year": "1872",
        "section": "Various Sections",
    },
    {
        "filename": "sample_consumer_protection_act_2019.txt",
        "source": "Consumer Protection Act, 2019 — Key Provisions",
        "doc_type": "Act / Statute",
        "jurisdiction": "India",
        "year": "2019",
        "section": "Various Sections",
    },
    {
        "filename": "sample_it_act_data_protection.txt",
        "source": "IT Act 2000 & DPDPA 2023 — Data Protection Overview",
        "doc_type": "Act / Statute",
        "jurisdiction": "India",
        "year": "2023",
        "section": "Various Sections",
    },
    {
        "filename": "sample_landlord_tenant_india.txt",
        "source": "Landlord-Tenant Laws in India — Overview",
        "doc_type": "Legal Overview",
        "jurisdiction": "India",
        "year": "2023",
        "section": "Various",
    },
]


def seed():
    store = get_vector_store(Config.VECTOR_STORE_PATH)
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
    total_chunks = 0

    for doc_meta in SAMPLE_DOCS:
        filepath = os.path.join(docs_dir, doc_meta["filename"])
        if not os.path.exists(filepath):
            print(f"  [SKIP] File not found: {filepath}")
            continue

        print(f"  Indexing: {doc_meta['source']}...")
        raw_text = extract_text_from_file(filepath)
        text = clean_text(raw_text)

        if not text.strip():
            print(f"  [SKIP] Empty text: {doc_meta['filename']}")
            continue

        metadata = {
            "source": doc_meta["source"],
            "doc_type": doc_meta["doc_type"],
            "jurisdiction": doc_meta["jurisdiction"],
            "year": doc_meta.get("year", ""),
            "section": doc_meta.get("section", ""),
            "filename": doc_meta["filename"],
        }

        chunks = chunk_text(
            text,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            metadata=metadata,
        )

        if not chunks:
            print(f"  [SKIP] No chunks produced for: {doc_meta['filename']}")
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, model_name=Config.EMBEDDING_MODEL)
        store.add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        print(f"  [OK] Indexed {len(chunks)} chunks from '{doc_meta['source']}'")

    print(f"\nDone! Total chunks in knowledge base: {store.total_chunks}")


if __name__ == "__main__":
    print("LawWise Knowledge Base Seeder")
    print("=" * 40)
    seed()
