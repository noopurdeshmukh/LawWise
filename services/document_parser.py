"""
Document Parser Service — validates and routes document extraction
"""
import os
import uuid
import logging
from werkzeug.utils import secure_filename
from rag.ingestion import extract_text_from_file, clean_text

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_upload(file_storage, upload_folder: str) -> dict:
    """
    Validate and save uploaded file.
    Returns dict with: filename, original_filename, filepath, file_type, file_size
    """
    original_filename = secure_filename(file_storage.filename)
    if not original_filename:
        raise ValueError("Invalid filename.")
    if not allowed_file(original_filename):
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    ext = original_filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, unique_filename)
    file_storage.save(filepath)

    file_size = os.path.getsize(filepath)
    if file_size > MAX_SIZE_BYTES:
        os.remove(filepath)
        raise ValueError(
            f"File too large. Maximum allowed size: {MAX_SIZE_BYTES // (1024*1024)} MB"
        )
    if file_size == 0:
        os.remove(filepath)
        raise ValueError("Uploaded file is empty.")

    return {
        "filename": unique_filename,
        "original_filename": original_filename,
        "filepath": filepath,
        "file_type": ext,
        "file_size": file_size,
    }


def parse_document(filepath: str) -> str:
    """Extract and clean text from a document."""
    raw_text = extract_text_from_file(filepath)
    return clean_text(raw_text)
