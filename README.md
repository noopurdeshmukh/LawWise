# LAW⚖WISE — AI-Powered Multi-Agent Legal Aid System

**"Understand the law. Know your rights. Make informed decisions."**

LawWise is a production-quality AI legal aid web application focused on making Indian legal information accessible to ordinary people. It combines a multi-agent AI architecture, Retrieval-Augmented Generation (RAG), and document intelligence to provide source-backed legal information.

---

## Problem Statement

Legal information is inaccessible to most people because of:
- Complex legal terminology
- Scattered information across large documents
- Difficulty finding relevant judgments
- Expensive legal consultations
- No easy way to understand contracts and notices

LawWise solves this with AI-powered tools that explain legal concepts in plain language with source citations.

---

## Features

- **Legal Q&A (RAG)** — Ask questions, get source-backed answers from the knowledge base
- **Document Analysis** — Upload and analyze legal notices, agreements, policies
- **Contract Review** — Identify risky clauses, obligations, key terms with risk ratings
- **Compliance Checker** — Compare documents against retrieved legal requirements
- **Case Research** — Find relevant Indian court judgments from the knowledge base
- **Knowledge Base** — Build and manage your indexed legal document corpus
- **Multi-Agent Architecture** — Intelligent routing to specialized AI agents
- **Source Citations** — Every answer shows which documents were used
- **Safety Guardrails** — Prominent disclaimers, no fabricated legal facts

---

## Architecture

```
USER QUERY
    ↓
LAWWISE ORCHESTRATOR (agents/orchestrator.py)
    ↓
INTENT DETECTION (keyword-based routing)
    ↓
SPECIALIZED AGENT
    ├── Q&A RAG Agent          → General legal questions
    ├── Contract Reviewer      → Contract analysis
    ├── Compliance Checker     → Regulatory compliance
    ├── Case Research Agent    → Court judgment research
    └── Document Analyzer      → Legal document extraction
    ↓
RAG RETRIEVAL (rag/retriever.py)
    ├── Query Embedding
    ├── Vector Store Search (FAISS / numpy cosine similarity)
    └── Top-K Chunks + Metadata
    ↓
GROQ LLM (services/groq_service.py)
    ├── System Prompt + Retrieved Context
    └── Structured JSON or Text Response
    ↓
STRUCTURED RESPONSE
    ├── Answer / Analysis
    ├── Source Citations
    └── Legal Disclaimer
```

---

## Multi-Agent Architecture

| Agent | File | Purpose |
|-------|------|---------|
| Orchestrator | `agents/orchestrator.py` | Routes queries to appropriate agents |
| Q&A RAG Agent | `agents/qa_rag_agent.py` | Source-backed legal Q&A |
| Contract Reviewer | `agents/contract_reviewer.py` | Contract risk analysis |
| Compliance Checker | `agents/compliance_checker.py` | Regulatory compliance checking |
| Case Research | `agents/case_research.py` | Court judgment retrieval |
| Document Analyzer | `agents/document_analyzer.py` | Legal document extraction |

---

## RAG Pipeline

```
1. Document Upload (PDF/DOCX/TXT)
        ↓
2. Text Extraction (rag/ingestion.py)
        ↓
3. Text Cleaning (remove noise, normalize)
        ↓
4. Chunking (rag/chunking.py) — 800 chars, 100 overlap
        ↓
5. Embedding (rag/embeddings.py) — TF-IDF hash or Sentence Transformers
        ↓
6. Vector Store (rag/vector_store.py) — FAISS or numpy cosine similarity
        ↓
7. Semantic Search (rag/retriever.py) — top-k with threshold
        ↓
8. Context + LLM → Source-backed Answer
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask |
| AI | Groq API (llama-3.3-70b-versatile) |
| RAG | FAISS (or numpy fallback), hash-TF-IDF or Sentence Transformers |
| Database | SQLite + SQLAlchemy |
| Frontend | HTML5, CSS3, Bootstrap 5, Vanilla JS |
| Document Parsing | PyPDF2, python-docx |
| Environment | python-dotenv |

---

## Project Structure

```
LawWise/
├── app.py                      # Flask app entry point
├── config.py                   # Configuration
├── routes.py                   # All Flask routes
├── requirements.txt
├── seed_knowledge_base.py      # Index sample documents
├── .env.example                # Environment template
│
├── agents/
│   ├── orchestrator.py         # Multi-agent orchestrator
│   ├── qa_rag_agent.py         # Q&A with RAG
│   ├── contract_reviewer.py    # Contract analysis
│   ├── compliance_checker.py   # Compliance checking
│   ├── case_research.py        # Case law research
│   └── document_analyzer.py    # Document extraction
│
├── rag/
│   ├── ingestion.py            # PDF/DOCX/TXT extraction
│   ├── chunking.py             # Text splitting
│   ├── embeddings.py           # Vector embeddings
│   ├── vector_store.py         # FAISS/numpy store
│   └── retriever.py            # Semantic search
│
├── services/
│   ├── groq_service.py         # Groq LLM client
│   ├── document_parser.py      # Upload validation
│   ├── citation_service.py     # Source citations
│   └── legal_service.py        # Document indexing
│
├── models/models.py            # SQLAlchemy models
├── database/db.py              # Database init
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html              # Landing page
│   ├── dashboard.html
│   ├── ask.html                # Chat interface
│   ├── upload.html             # Document management
│   ├── document.html           # Document viewer
│   ├── contract_review.html
│   ├── compliance.html
│   ├── cases.html
│   ├── knowledge_base.html
│   └── settings.html
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── data/
│   ├── documents/              # Uploaded files
│   └── vector_store/          # FAISS index files
│
└── tests/
    └── test_lawwise.py         # Test suite (29 tests)
```

---

## Installation (Windows)

### 1. Prerequisites

- Python 3.9+ (64-bit recommended for FAISS/torch support)
- Groq API key (free at https://console.groq.com)

### 2. Clone / Navigate to Project

```powershell
cd LawWise
```

### 3. Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note for 32-bit Python users:** faiss-cpu and torch are not available for 32-bit Python. LawWise includes a numpy-based TF-IDF cosine similarity fallback that works without these packages. For best semantic search quality, use 64-bit Python.

### 5. Configure Environment

```powershell
copy .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
SECRET_KEY=your_random_secret_key
FLASK_ENV=development
```

Get your free Groq API key at: https://console.groq.com

### 6. Seed the Knowledge Base

```powershell
python seed_knowledge_base.py
```

This indexes the sample legal documents (Indian Contract Act, Consumer Protection Act, IT Act/DPDPA, Tenancy Laws).

### 7. Run the Application

```powershell
python app.py
```

Open http://localhost:5000 in your browser.

---

## Adding Legal Documents

1. Navigate to **My Documents** or **Knowledge Base** in the app
2. Upload PDF, DOCX, or TXT files
3. Select jurisdiction and document type
4. LawWise automatically extracts, chunks, embeds, and indexes the document
5. Ask questions — the document will now appear in search results

Or use the Python API:

```python
from services.legal_service import index_document

index_document(
    filepath="path/to/act.pdf",
    metadata={
        "source": "Companies Act, 2013",
        "doc_type": "Act / Statute",
        "jurisdiction": "India",
        "year": "2013",
    }
)
```

---

## Running Tests

```powershell
python -m pytest tests/test_lawwise.py -v
```

All 29 tests should pass:
- Chunking tests
- Text extraction tests
- Document parser / upload validation tests
- Flask route tests (all 13 routes)
- Groq service error handling tests
- Vector store tests

---

## Security

- Groq API key loaded from `.env` only — never sent to frontend
- File type validation (PDF, DOCX, TXT only)
- File size limit (configurable, default 10 MB)
- Secure filename generation (UUID prefix)
- Safe error messages (no stack traces to users)
- SQLAlchemy ORM (no raw SQL injection risk)

---

## Legal Disclaimer

LawWise provides general legal information for educational purposes only. It is not a substitute for advice from a qualified lawyer. Always consult a licensed legal professional for your specific situation.

LawWise never:
- Claims to be a lawyer
- Guarantees legal outcomes
- Fabricates case names, citations, or legal holdings
- Certifies legal compliance

---

## Known Limitations

1. **32-bit Python**: faiss-cpu and sentence-transformers (torch) cannot be installed. The numpy TF-IDF fallback works but provides weaker semantic similarity than neural embeddings.
2. **Knowledge base quality**: The app is only as good as the documents you index. Start with verified legal texts.
3. **No OCR**: Scanned PDFs need digital text extraction. Add pytesseract for OCR support.
4. **English only**: Hindi and regional Indian languages are future scope.
5. **Synchronous indexing**: Large documents block the request during indexing. Add Celery for async processing in production.

---

## Future Improvements

- OCR support (pytesseract)
- 64-bit Python for full Sentence Transformers + FAISS support
- Hindi and regional Indian language support
- Voice assistant interface
- Expanded Indian legal corpus
- Lawyer consultation booking
- PDF legal report export
- Admin panel with analytics
- Cloud vector database (Pinecone, Weaviate)
- Docker deployment
- Celery for async document processing
