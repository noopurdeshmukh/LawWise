"""
LawWise Database Models
"""
from datetime import datetime
from database.db import db


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)
    file_size = db.Column(db.Integer, default=0)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    jurisdiction = db.Column(db.String(100), default="India")
    doc_type = db.Column(db.String(100), default="General")
    status = db.Column(db.String(50), default="uploaded")  # uploaded, indexed, analyzed
    chunk_count = db.Column(db.Integer, default=0)
    analysis_summary = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "upload_date": self.upload_date.strftime("%Y-%m-%d %H:%M"),
            "jurisdiction": self.jurisdiction,
            "doc_type": self.doc_type,
            "status": self.status,
            "chunk_count": self.chunk_count,
        }


class ChatHistory(db.Model):
    __tablename__ = "chat_history"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user / assistant
    content = db.Column(db.Text, nullable=False)
    agent_used = db.Column(db.String(100), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sources_json = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "agent_used": self.agent_used,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M"),
        }


class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=True)
    analysis_type = db.Column(db.String(100), nullable=False)
    result_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    document = db.relationship("Document", backref="analyses")

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "analysis_type": self.analysis_type,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }
