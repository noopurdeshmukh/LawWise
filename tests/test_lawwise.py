"""
LawWise Tests — Core functionality
"""
import os
import sys
import pytest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()


# ── Chunking Tests ───────────────────────────────────────────

class TestChunking:
    def test_basic_chunking(self):
        from rag.chunking import chunk_text
        text = "This is a test sentence. " * 50
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)

    def test_chunk_metadata(self):
        from rag.chunking import chunk_text
        meta = {"source": "test_doc", "doc_type": "Test"}
        text = "Contract law is important. " * 40
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50, metadata=meta)
        assert all(c["metadata"]["source"] == "test_doc" for c in chunks)
        assert all(c["metadata"]["doc_type"] == "Test" for c in chunks)

    def test_empty_text(self):
        from rag.chunking import chunk_text
        chunks = chunk_text("", chunk_size=200, chunk_overlap=50)
        assert chunks == []

    def test_short_text(self):
        from rag.chunking import chunk_text
        chunks = chunk_text("Short text.", chunk_size=200, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."


# ── Text Extraction Tests ─────────────────────────────────────

class TestIngestion:
    def test_txt_extraction(self, tmp_path):
        from rag.ingestion import extract_text_from_file, clean_text
        f = tmp_path / "test.txt"
        f.write_text("Hello   world.\nThis is a test.")
        text = extract_text_from_file(str(f))
        assert "Hello" in text
        assert "test" in text

    def test_clean_text(self):
        from rag.ingestion import clean_text
        dirty = "Hello   \t world  \n\n  test\x00"
        cleaned = clean_text(dirty)
        assert "\x00" not in cleaned
        assert "  " not in cleaned  # no double spaces

    def test_unsupported_extension(self, tmp_path):
        from rag.ingestion import extract_text_from_file
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError):
            extract_text_from_file(str(f))


# ── Document Parser Tests ─────────────────────────────────────

class TestDocumentParser:
    def test_allowed_file(self):
        from services.document_parser import allowed_file
        assert allowed_file("contract.pdf") is True
        assert allowed_file("doc.docx") is True
        assert allowed_file("text.txt") is True
        assert allowed_file("malware.exe") is False
        assert allowed_file("script.js") is False
        assert allowed_file("noextension") is False

    def test_save_upload_invalid_type(self, tmp_path):
        from services.document_parser import save_upload
        from werkzeug.datastructures import FileStorage
        import io
        fs = FileStorage(
            stream=io.BytesIO(b"content"),
            filename="test.exe",
        )
        with pytest.raises(ValueError, match="Unsupported file type"):
            save_upload(fs, str(tmp_path))

    def test_save_upload_empty_filename(self, tmp_path):
        from services.document_parser import save_upload
        from werkzeug.datastructures import FileStorage
        import io
        fs = FileStorage(stream=io.BytesIO(b"content"), filename="")
        with pytest.raises(ValueError):
            save_upload(fs, str(tmp_path))


# ── Flask Routes Tests ─────────────────────────────────────────

class TestFlaskRoutes:
    @pytest.fixture
    def client(self):
        from app import create_app
        from config import Config
        import tempfile

        class TestConfig(Config):
            TESTING = True
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            WTF_CSRF_ENABLED = False
            UPLOAD_FOLDER = tempfile.mkdtemp()

        app = create_app(TestConfig)
        with app.test_client() as client:
            yield client

    def test_home_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"LawWise" in resp.data

    def test_dashboard(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_ask_get(self, client):
        resp = client.get("/ask")
        assert resp.status_code == 200

    def test_upload_get(self, client):
        resp = client.get("/upload")
        assert resp.status_code == 200

    def test_contract_review_get(self, client):
        resp = client.get("/contract-review")
        assert resp.status_code == 200

    def test_compliance_get(self, client):
        resp = client.get("/compliance")
        assert resp.status_code == 200

    def test_cases_get(self, client):
        resp = client.get("/cases")
        assert resp.status_code == 200

    def test_knowledge_base_get(self, client):
        resp = client.get("/knowledge-base")
        assert resp.status_code == 200

    def test_settings_get(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "status" in data

    def test_ask_empty_query(self, client):
        resp = client.post("/ask", json={"query": ""})
        assert resp.status_code == 400

    def test_upload_no_file(self, client):
        resp = client.post("/upload", data={})
        assert resp.status_code == 400

    def test_upload_invalid_type(self, client):
        import io
        data = {"file": (io.BytesIO(b"content"), "test.exe")}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_case_research_empty(self, client):
        resp = client.post("/case-research", json={"query": ""})
        assert resp.status_code == 400

    def test_404(self, client):
        resp = client.get("/nonexistent-page")
        assert resp.status_code == 404


# ── Groq Service Tests ────────────────────────────────────────

class TestGroqService:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        from services import groq_service
        result = groq_service.call_llm("system", "user prompt")
        assert "api key" in result.lower() or "GROQ_API_KEY" in result or "Invalid" in result

    def test_call_llm_json_invalid_response(self, monkeypatch):
        """Test that call_llm_json handles non-JSON response gracefully."""
        from services import groq_service
        # Mock call_llm to return plain text
        monkeypatch.setattr(groq_service, "call_llm", lambda *a, **kw: "This is plain text not JSON")
        result = groq_service.call_llm_json("system", "user")
        assert result is None


# ── Vector Store Tests ─────────────────────────────────────────

class TestVectorStore:
    def test_empty_store(self, tmp_path):
        from rag.vector_store import VectorStore
        store = VectorStore(str(tmp_path))
        assert store.total_chunks == 0

    def test_search_empty_store(self, tmp_path):
        from rag.vector_store import VectorStore
        import numpy as np
        store = VectorStore(str(tmp_path))
        result = store.search(np.zeros((1, 384), dtype=np.float32))
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
