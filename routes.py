"""
LawWise Flask Application Routes
"""
import os
import json
import uuid
import logging
from datetime import datetime
from flask import (
    Blueprint, render_template, request, jsonify, redirect,
    url_for, session, flash, current_app
)

from database.db import db
from models.models import Document, ChatHistory, AnalysisResult
from services.document_parser import save_upload, parse_document, allowed_file
from services.legal_service import index_document, get_knowledge_base_stats
from services.citation_service import get_disclaimer
from agents.orchestrator import route_to_agent

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────

@main.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

@main.route("/dashboard")
def dashboard():
    recent_docs = Document.query.order_by(Document.upload_date.desc()).limit(5).all()
    recent_chats = (
        ChatHistory.query
        .filter_by(session_id=get_session_id(), role="user")
        .order_by(ChatHistory.timestamp.desc())
        .limit(5)
        .all()
    )
    stats = get_knowledge_base_stats()
    return render_template(
        "dashboard.html",
        recent_docs=recent_docs,
        recent_chats=recent_chats,
        stats=stats,
    )


# ─────────────────────────────────────────────
# ASK LAWWISE (Chat)
# ─────────────────────────────────────────────

@main.route("/ask", methods=["GET"])
def ask():
    sid = get_session_id()
    history = (
        ChatHistory.query
        .filter_by(session_id=sid)
        .order_by(ChatHistory.timestamp.asc())
        .limit(50)
        .all()
    )
    return render_template("ask.html", history=history, disclaimer=get_disclaimer())


@main.route("/ask", methods=["POST"])
def ask_post():
    sid = get_session_id()
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    jurisdiction = data.get("jurisdiction", "India")

    if not query:
        return jsonify({"error": "Please enter a question."}), 400

    if len(query) > 2000:
        return jsonify({"error": "Question too long. Please limit to 2000 characters."}), 400

    # Save user message
    user_msg = ChatHistory(
        session_id=sid, role="user", content=query
    )
    db.session.add(user_msg)
    db.session.commit()

    # Route through orchestrator
    result = route_to_agent(query, jurisdiction=jurisdiction)

    answer = result.get("answer") or result.get("raw_analysis", "Unable to generate a response.")
    agent_used = result.get("primary_agent", "qa_rag")
    sources = result.get("sources", [])

    # Save assistant message
    assistant_msg = ChatHistory(
        session_id=sid,
        role="assistant",
        content=answer,
        agent_used=agent_used,
        sources_json=json.dumps(sources),
    )
    db.session.add(assistant_msg)
    db.session.commit()

    return jsonify({
        "answer": answer,
        "agent_used": agent_used,
        "agent_steps": result.get("agent_steps", []),
        "sources": sources,
        "disclaimer": result.get("disclaimer", get_disclaimer()),
    })


@main.route("/ask/history/clear", methods=["POST"])
def clear_chat():
    sid = get_session_id()
    ChatHistory.query.filter_by(session_id=sid).delete()
    db.session.commit()
    return jsonify({"success": True})


# ─────────────────────────────────────────────
# DOCUMENT UPLOAD & MANAGEMENT
# ─────────────────────────────────────────────

@main.route("/upload", methods=["GET"])
def upload():
    docs = Document.query.order_by(Document.upload_date.desc()).all()
    return render_template("upload.html", documents=docs)


@main.route("/upload", methods=["POST"])
def upload_post():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    jurisdiction = request.form.get("jurisdiction", "India")
    doc_type = request.form.get("doc_type", "General")

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        file_info = save_upload(file, current_app.config["UPLOAD_FOLDER"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Save to DB
    doc = Document(
        filename=file_info["filename"],
        original_filename=file_info["original_filename"],
        file_type=file_info["file_type"],
        file_size=file_info["file_size"],
        jurisdiction=jurisdiction,
        doc_type=doc_type,
        status="uploaded",
    )
    db.session.add(doc)
    db.session.commit()

    # Index document in background (synchronous for now)
    try:
        metadata = {
            "source": file_info["original_filename"],
            "doc_type": doc_type,
            "jurisdiction": jurisdiction,
            "upload_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }
        chunk_count = index_document(file_info["filepath"], metadata)
        doc.status = "indexed"
        doc.chunk_count = chunk_count
        db.session.commit()
        return jsonify({
            "success": True,
            "document_id": doc.id,
            "filename": file_info["original_filename"],
            "chunk_count": chunk_count,
            "message": f"Document uploaded and indexed ({chunk_count} chunks).",
        })
    except Exception as e:
        doc.status = "error"
        db.session.commit()
        logger.error(f"Indexing error: {e}")
        return jsonify({
            "success": True,
            "document_id": doc.id,
            "filename": file_info["original_filename"],
            "warning": f"Document saved but indexing failed: {str(e)}",
        })


@main.route("/document/<int:doc_id>", methods=["GET"])
def document_view(doc_id):
    doc = Document.query.get_or_404(doc_id)
    # Get analysis results
    analyses = AnalysisResult.query.filter_by(document_id=doc_id).order_by(AnalysisResult.created_at.desc()).all()
    return render_template("document.html", doc=doc, analyses=analyses)


@main.route("/document/<int:doc_id>/delete", methods=["POST"])
def document_delete(doc_id):
    doc = Document.query.get_or_404(doc_id)
    # Remove from vector store
    try:
        from rag.vector_store import get_vector_store
        store = get_vector_store()
        store.delete_by_document(doc.original_filename)
    except Exception as e:
        logger.warning(f"Could not remove from vector store: {e}")
    # Remove file
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], doc.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"success": True})


# ─────────────────────────────────────────────
# ANALYZE DOCUMENT
# ─────────────────────────────────────────────

@main.route("/analyze-document", methods=["POST"])
def analyze_document():
    doc_id = request.form.get("document_id") or (request.get_json() or {}).get("document_id")
    doc_text = request.form.get("document_text") or (request.get_json() or {}).get("document_text", "")

    doc = None
    filename = "Document"

    if doc_id:
        doc = Document.query.get(doc_id)
        if doc:
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], doc.filename)
            try:
                doc_text = parse_document(filepath)
                filename = doc.original_filename
            except Exception as e:
                return jsonify({"error": f"Could not read document: {str(e)}"}), 500

    if not doc_text:
        return jsonify({"error": "No document text provided."}), 400

    import agents.document_analyzer as analyzer
    result = analyzer.run(doc_text, filename)

    if doc:
        analysis = AnalysisResult(
            document_id=doc.id,
            analysis_type="document_analysis",
            result_json=json.dumps(result),
        )
        db.session.add(analysis)
        doc.status = "analyzed"
        db.session.commit()

    return jsonify(result)


# ─────────────────────────────────────────────
# CONTRACT REVIEW
# ─────────────────────────────────────────────

@main.route("/contract-review", methods=["GET"])
def contract_review():
    docs = Document.query.order_by(Document.upload_date.desc()).all()
    return render_template("contract_review.html", documents=docs)


@main.route("/contract-review", methods=["POST"])
def contract_review_post():
    data = request.get_json() or {}
    doc_id = data.get("document_id")
    doc_text = data.get("document_text", "")

    doc = None
    filename = "Contract"

    if doc_id:
        doc = Document.query.get(doc_id)
        if doc:
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], doc.filename)
            try:
                doc_text = parse_document(filepath)
                filename = doc.original_filename
            except Exception as e:
                return jsonify({"error": f"Could not read document: {str(e)}"}), 500

    if not doc_text or len(doc_text.strip()) < 100:
        return jsonify({"error": "Please provide a contract with sufficient text."}), 400

    import agents.contract_reviewer as reviewer
    result = reviewer.run(doc_text, filename)

    if doc:
        analysis = AnalysisResult(
            document_id=doc.id,
            analysis_type="contract_review",
            result_json=json.dumps(result),
        )
        db.session.add(analysis)
        db.session.commit()

    return jsonify(result)


# ─────────────────────────────────────────────
# COMPLIANCE CHECK
# ─────────────────────────────────────────────

@main.route("/compliance", methods=["GET"])
def compliance():
    docs = Document.query.order_by(Document.upload_date.desc()).all()
    return render_template("compliance.html", documents=docs)


@main.route("/compliance-check", methods=["POST"])
def compliance_check():
    data = request.get_json() or {}
    doc_id = data.get("document_id")
    doc_text = data.get("document_text", "")
    business_description = data.get("business_description", "")
    jurisdiction = data.get("jurisdiction", "India")

    if doc_id:
        doc = Document.query.get(doc_id)
        if doc:
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], doc.filename)
            try:
                doc_text = parse_document(filepath)
            except Exception as e:
                return jsonify({"error": f"Could not read document: {str(e)}"}), 500

    import agents.compliance_checker as checker
    result = checker.run(doc_text, business_description, jurisdiction)
    return jsonify(result)


# ─────────────────────────────────────────────
# CASE RESEARCH
# ─────────────────────────────────────────────

@main.route("/cases", methods=["GET"])
def cases():
    return render_template("cases.html")


@main.route("/case-research", methods=["POST"])
def case_research():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    jurisdiction = data.get("jurisdiction", "India")

    if not query:
        return jsonify({"error": "Please enter a research query."}), 400

    import agents.case_research as researcher
    result = researcher.run(query, jurisdiction)
    return jsonify(result)


# ─────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────

@main.route("/knowledge-base", methods=["GET"])
def knowledge_base():
    stats = get_knowledge_base_stats()
    docs = Document.query.order_by(Document.upload_date.desc()).all()
    return render_template("knowledge_base.html", stats=stats, documents=docs)


@main.route("/api/knowledge-base/stats", methods=["GET"])
def kb_stats():
    stats = get_knowledge_base_stats()
    return jsonify(stats)


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

@main.route("/settings", methods=["GET"])
def settings():
    groq_configured = bool(os.environ.get("GROQ_API_KEY", ""))
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    stats = get_knowledge_base_stats()
    return render_template(
        "settings.html",
        groq_configured=groq_configured,
        model=model,
        stats=stats,
    )


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@main.route("/api/health", methods=["GET"])
def health():
    groq_ok = bool(os.environ.get("GROQ_API_KEY", ""))
    stats = get_knowledge_base_stats()
    return jsonify({
        "status": "ok",
        "groq_configured": groq_ok,
        "knowledge_base": stats,
        "timestamp": datetime.utcnow().isoformat(),
    })


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────

@main.app_errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@main.app_errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size exceeded."}), 413


@main.app_errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal server error."), 500
