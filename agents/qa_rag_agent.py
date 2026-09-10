"""
Q&A RAG Agent — answers legal questions using retrieved documents
"""
import logging
from typing import Dict, Any, List

from rag.retriever import retrieve, format_context_for_llm, get_source_citations
from services.groq_service import call_llm
from services.citation_service import get_disclaimer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are LawWise Assistant, an expert AI legal information assistant focused on Indian law.

You answer legal questions using ONLY the provided source documents.
You provide clear, structured answers with the following sections:

### Simple Explanation
Explain the concept in plain language that a non-lawyer can understand.

### Legal Context
Explain the relevant law, statute, or legal principle from the sources.

### Why It Matters
Explain the practical implications for the user.

### Important Note
Always include a disclaimer that LawWise provides general information only, not legal advice.

RULES:
- Only use information from the provided sources.
- If the sources do not contain enough information, say so clearly.
- Never fabricate legal facts, case names, citations, or statutes.
- Never guarantee legal outcomes.
- Identify the jurisdiction when relevant.
- Be clear about uncertainty."""


def run(query: str, document_text: str = "", jurisdiction: str = "India") -> Dict[str, Any]:
    """
    Run the Q&A RAG Agent.
    
    Args:
        query: User's legal question
        document_text: Optional uploaded document text for context
        jurisdiction: Legal jurisdiction
    
    Returns:
        dict with answer, sources, disclaimer
    """
    # Retrieve relevant chunks
    chunks = retrieve(
        query,
        top_k=5,
        similarity_threshold=0.05,
        filters={"jurisdiction": jurisdiction} if jurisdiction else None,
    )

    # If no jurisdiction-filtered results, try without filter
    if not chunks and jurisdiction:
        chunks = retrieve(query, top_k=5, similarity_threshold=0.05)

    context = format_context_for_llm(chunks)
    citations = get_source_citations(chunks)

    # Build prompt
    doc_context = ""
    if document_text:
        doc_context = f"\n\nUSER-PROVIDED DOCUMENT EXCERPT:\n{document_text[:3000]}"

    if not context and not doc_context:
        return {
            "answer": (
                "I couldn't find enough relevant material in the available LawWise legal knowledge base "
                "to provide a reliable source-backed answer to your question.\n\n"
                "**What you can do:**\n"
                "- Upload relevant legal documents to the knowledge base\n"
                "- Try rephrasing your question\n"
                "- Consult a qualified lawyer for authoritative guidance"
            ),
            "sources": [],
            "agent": "qa_rag",
            "disclaimer": get_disclaimer(),
            "has_sources": False,
        }

    user_prompt = f"""LEGAL QUESTION: {query}

JURISDICTION: {jurisdiction}

RETRIEVED LEGAL SOURCES:
{context if context else "No indexed sources available for this query."}
{doc_context}

Please answer the question using the above sources. Structure your response with:
### Simple Explanation
### Legal Context  
### Why It Matters
### Important Note

If the sources don't have enough information, clearly state that."""

    answer = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=2000)

    return {
        "answer": answer,
        "sources": citations,
        "agent": "qa_rag",
        "disclaimer": get_disclaimer(),
        "has_sources": len(citations) > 0,
        "query": query,
    }
