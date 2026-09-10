"""
Document Analyzer Agent — extracts key information from legal documents
"""
import json
import logging
from typing import Dict, Any

from rag.retriever import retrieve, get_source_citations
from services.groq_service import call_llm, call_llm_json
from services.citation_service import get_disclaimer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are LawWise Document Analyzer, a specialized AI for analyzing legal documents.

You extract and explain key information from legal documents in plain language.
You analyze: legal notices, contracts, agreements, policies, terms and conditions, government notices.

ANALYSIS APPROACH:
1. Identify what type of document this is
2. Extract key structured information
3. Explain in plain language what the document means
4. Identify obligations, rights, deadlines, and financial terms
5. Highlight potential risks or missing information
6. Provide a clear plain-language summary

RULES:
- Be accurate about what you find in the document
- Note if key information appears to be missing
- Do not interpret ambiguous clauses with false certainty
- Always recommend professional legal advice for consequential matters
- Focus on practical implications for the reader

Return valid JSON only."""

ANALYZER_SCHEMA = {
    "document_type": "Type of legal document",
    "summary": "Plain language summary of the document",
    "parties": [{"role": "...", "name": "..."}],
    "key_dates": [{"event": "...", "date": "..."}],
    "obligations": [{"party": "...", "obligation": "...", "deadline": "..."}],
    "rights": [{"party": "...", "right": "..."}],
    "financial_terms": [{"description": "...", "amount": "...", "condition": "..."}],
    "important_clauses": [
        {
            "title": "Clause title",
            "text": "Clause text",
            "category": "risk|obligation|payment|deadline|termination|confidentiality|dispute",
            "explanation": "Plain language explanation",
            "attention_level": "high|medium|low"
        }
    ],
    "potential_risks": [{"risk": "...", "explanation": "..."}],
    "missing_information": ["What seems to be missing"],
    "action_items": ["What the reader should do or consider"],
    "questions_for_lawyer": ["Question 1", "Question 2"],
    "disclaimer": "LawWise disclaimer"
}


def run(document_text: str, filename: str = "Document", doc_type_hint: str = "") -> Dict[str, Any]:
    """
    Run the Document Analyzer Agent.
    
    Args:
        document_text: Text of the document to analyze
        filename: Name of the document
        doc_type_hint: Optional hint about document type
    
    Returns:
        dict with document analysis
    """
    if not document_text or len(document_text.strip()) < 50:
        return {
            "error": "Document text is too short or empty for analysis.",
            "agent": "document_analyzer",
        }

    # Get relevant legal context from knowledge base
    search_q = doc_type_hint if doc_type_hint else "legal document analysis"
    chunks = retrieve(search_q, top_k=3, similarity_threshold=0.05)
    legal_context = "\n\n".join([c["text"] for c in chunks[:2]]) if chunks else ""

    user_prompt = f"""Analyze the following legal document and extract key information.

DOCUMENT NAME: {filename}
{f"DOCUMENT TYPE HINT: {doc_type_hint}" if doc_type_hint else ""}

DOCUMENT TEXT:
{document_text[:7000]}

{"RELEVANT LEGAL CONTEXT:" + chr(10) + legal_context[:1500] if legal_context else ""}

Return ONLY valid JSON matching this structure:
{json.dumps(ANALYZER_SCHEMA, indent=2)}

Extract all significant clauses under important_clauses (minimum 4).
For attention_level use: high (significant impact), medium (notable), low (standard/routine).
Be specific and accurate — only report what is actually in the document."""

    result = call_llm_json(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=4000)

    if result is None:
        text_result = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=3000)
        return {
            "raw_analysis": text_result,
            "agent": "document_analyzer",
            "filename": filename,
            "disclaimer": get_disclaimer(),
            "json_failed": True,
        }

    result["agent"] = "document_analyzer"
    result["filename"] = filename
    result["disclaimer"] = get_disclaimer()
    result["sources"] = get_source_citations(chunks) if chunks else []
    return result
