"""
Compliance Checker Agent — checks documents/practices against legal requirements
"""
import json
import logging
from typing import Dict, Any

from rag.retriever import retrieve, get_source_citations, format_context_for_llm
from services.groq_service import call_llm, call_llm_json
from services.citation_service import get_disclaimer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are LawWise Compliance Checker, a specialized AI assistant for compliance analysis.

You compare documents or described business practices against relevant legal and regulatory requirements
retrieved from the LawWise knowledge base.

COMPLIANCE STATUS CODES:
- ✅ Appears Aligned: Based on available sources, this appears to meet the requirement
- ⚠️ Needs Review: Unclear from available information; professional verification recommended
- ❌ Potential Issue: Available sources suggest a potential compliance concern
- ❓ Insufficient Information: Not enough information to assess

CRITICAL RULES:
- NEVER say "You are legally compliant" with certainty
- Always say "Based on available sources, this appears aligned with..." 
- "Potential compliance issue identified. Professional verification is recommended."
- Only reference requirements found in the retrieved sources
- Do not invent regulatory requirements
- Always recommend professional legal/compliance advice

Return valid JSON only."""

COMPLIANCE_SCHEMA = {
    "summary": "Overall compliance overview",
    "document_type": "Type of document analyzed",
    "jurisdiction": "India",
    "checks": [
        {
            "requirement": "The legal/regulatory requirement",
            "current_situation": "What the document says",
            "status": "aligned|needs_review|potential_issue|insufficient_info",
            "explanation": "Detailed explanation",
            "source": "Source document or regulation",
            "recommendation": "What action to take"
        }
    ],
    "overall_assessment": "Summary of compliance posture",
    "key_concerns": ["Concern 1", "Concern 2"],
    "recommended_actions": ["Action 1", "Action 2"],
    "disclaimer": "LawWise disclaimer"
}


def run(document_text: str, business_description: str = "", jurisdiction: str = "India") -> Dict[str, Any]:
    """
    Run the Compliance Checker Agent.
    
    Args:
        document_text: Document text to check
        business_description: Optional description of the business context
        jurisdiction: Legal jurisdiction
    
    Returns:
        dict with compliance analysis
    """
    # Retrieve relevant compliance/regulatory documents
    search_query = f"compliance requirements regulations obligations {business_description[:100] if business_description else 'legal requirements'}"
    chunks = retrieve(search_query, top_k=5, similarity_threshold=0.05)
    
    if not chunks:
        chunks = retrieve("legal requirements India compliance", top_k=5, similarity_threshold=0.05)

    context = format_context_for_llm(chunks)
    citations = get_source_citations(chunks)

    user_prompt = f"""Perform a compliance analysis on the following document/practice.

JURISDICTION: {jurisdiction}
{f"BUSINESS CONTEXT: {business_description}" if business_description else ""}

DOCUMENT TO ANALYZE:
{document_text[:6000] if document_text else "No document provided — analyze based on business description above."}

RETRIEVED LEGAL/REGULATORY REQUIREMENTS FROM KNOWLEDGE BASE:
{context if context else "Limited regulatory sources available in the knowledge base."}

Return ONLY valid JSON matching this structure:
{json.dumps(COMPLIANCE_SCHEMA, indent=2)}

Identify at least 3 compliance dimensions to check.
Be specific about which requirements from the knowledge base apply.
If knowledge base sources are limited, note this in the summary."""

    result = call_llm_json(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=3500)

    if result is None:
        text_result = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=2500)
        return {
            "raw_analysis": text_result,
            "agent": "compliance_checker",
            "disclaimer": get_disclaimer(),
            "json_failed": True,
            "sources": citations,
        }

    result["agent"] = "compliance_checker"
    result["disclaimer"] = get_disclaimer()
    result["sources"] = citations
    return result
