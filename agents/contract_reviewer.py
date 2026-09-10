"""
Contract Reviewer Agent — analyzes contracts for risks, obligations, and key terms
"""
import json
import logging
from typing import Dict, Any

from rag.retriever import retrieve, get_source_citations
from services.groq_service import call_llm, call_llm_json
from services.citation_service import get_disclaimer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are LawWise Contract Reviewer, a specialized AI assistant for analyzing legal contracts.

You analyze contracts thoroughly and identify potential risks, obligations, and key terms.

For each finding you MUST use this risk language:
- 🔴 High Attention: Clauses that significantly affect the user's rights or create substantial obligations
- 🟠 Medium Attention: Clauses that may need clarification or negotiation
- 🟢 Low Attention: Standard clauses that appear reasonable but should be noted

IMPORTANT RULES:
- Do NOT call any clause "illegal" unless it clearly violates a stated law in the sources
- Use "potential concern", "may require review", "could create a risk depending on circumstances"
- Never guarantee legal outcomes
- Always recommend consulting a qualified lawyer for important matters
- Focus on Indian law context unless specified otherwise

Return a JSON response with the structure below. Ensure valid JSON only."""

CONTRACT_REVIEW_SCHEMA = {
    "contract_overview": "Brief description of the contract",
    "parties": ["Party 1", "Party 2"],
    "main_purpose": "What the contract is for",
    "important_dates": [{"event": "...", "date": "..."}],
    "payment_terms": "Payment details if any",
    "major_obligations": [{"party": "...", "obligation": "..."}],
    "termination": "How the contract can be terminated",
    "liability": "Liability provisions",
    "confidentiality": "Confidentiality terms",
    "dispute_resolution": "How disputes are resolved",
    "governing_law": "Which law governs",
    "risks": [
        {
            "clause": "Clause text or title",
            "risk_level": "high|medium|low",
            "explanation": "What this means",
            "why_it_matters": "Why this is important",
            "potential_concern": "The specific concern",
            "question_for_lawyer": "What to ask your lawyer"
        }
    ],
    "questions_for_lawyer": ["Question 1", "Question 2"],
    "disclaimer": "LawWise disclaimer"
}


def run(document_text: str, filename: str = "Contract") -> Dict[str, Any]:
    """
    Run the Contract Reviewer Agent.
    
    Args:
        document_text: Full text of the contract
        filename: Name of the contract file
    
    Returns:
        dict with contract analysis
    """
    if not document_text or len(document_text.strip()) < 100:
        return {
            "error": "Contract text is too short or empty for meaningful analysis.",
            "agent": "contract_reviewer",
        }

    # Get relevant legal sources for context
    chunks = retrieve("contract clauses obligations liability indemnity", top_k=3, similarity_threshold=0.05)
    legal_context = "\n\n".join([c["text"] for c in chunks[:3]]) if chunks else ""

    user_prompt = f"""Analyze the following contract and return a comprehensive JSON review.

CONTRACT NAME: {filename}

CONTRACT TEXT:
{document_text[:8000]}

{"RELEVANT LEGAL CONTEXT FROM KNOWLEDGE BASE:" + chr(10) + legal_context if legal_context else ""}

Return ONLY valid JSON matching this structure:
{json.dumps(CONTRACT_REVIEW_SCHEMA, indent=2)}

For the risks array, identify ALL significant clauses (minimum 3, maximum 15).
Be specific about which part of the contract each risk refers to.
Use exact quotes from the contract where possible in the 'clause' field."""

    result = call_llm_json(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=4000)

    if result is None:
        # Fallback to text-based analysis
        text_result = call_llm(SYSTEM_PROMPT, user_prompt + "\n\nNote: Return as structured text if JSON fails.", temperature=0.2, max_tokens=3000)
        return {
            "raw_analysis": text_result,
            "agent": "contract_reviewer",
            "filename": filename,
            "disclaimer": get_disclaimer(),
            "json_failed": True,
        }

    result["agent"] = "contract_reviewer"
    result["filename"] = filename
    result["disclaimer"] = get_disclaimer()
    result["sources"] = get_source_citations(chunks) if chunks else []
    return result
