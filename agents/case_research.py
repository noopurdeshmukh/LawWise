"""
Case Research Agent — retrieves relevant Indian court judgments from the knowledge base
"""
import logging
from typing import Dict, Any

from rag.retriever import retrieve, get_source_citations, format_context_for_llm
from services.groq_service import call_llm
from services.citation_service import get_disclaimer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are LawWise Case Research Assistant, specialized in Indian case law research.

You help users find relevant court judgments and legal precedents from the LawWise knowledge base.

For each relevant case found in the sources, provide:
- Case name (ONLY from the sources — never invent)
- Court
- Year
- Legal issue
- Relevant finding/holding
- Why it may be relevant to the user's query

CRITICAL RULES — STRICTLY ENFORCED:
- NEVER fabricate case names, case numbers, citations, judges, dates, or legal holdings
- ONLY reference cases that appear in the provided source documents
- If no relevant cases are found in the knowledge base, clearly state:
  "No sufficiently relevant case was found in the available LawWise knowledge base."
- Do NOT fill gaps with invented "illustrative" cases
- Distinguish between cases directly found in sources vs. general legal principles
- Always note that case law research requires professional legal expertise"""


def run(query: str, jurisdiction: str = "India") -> Dict[str, Any]:
    """
    Run the Case Research Agent.
    
    Args:
        query: Research query
        jurisdiction: Legal jurisdiction (default India)
    
    Returns:
        dict with case research results
    """
    # Search for relevant judgments
    search_queries = [
        f"{query} judgment court India",
        f"{query} case law Supreme Court High Court",
        query,
    ]

    all_chunks = []
    seen_texts = set()

    for sq in search_queries:
        chunks = retrieve(sq, top_k=4, similarity_threshold=0.05)
        for c in chunks:
            if c["text"] not in seen_texts:
                seen_texts.add(c["text"])
                all_chunks.append(c)
        if len(all_chunks) >= 8:
            break

    # Filter for case/judgment related chunks
    case_chunks = [
        c for c in all_chunks
        if any(kw in c.get("metadata", {}).get("doc_type", "").lower()
               for kw in ["judgment", "case", "order", "verdict"])
    ]
    if not case_chunks:
        case_chunks = all_chunks  # use all if no explicit case docs

    context = format_context_for_llm(case_chunks[:6])
    citations = get_source_citations(case_chunks[:6])

    if not context:
        return {
            "answer": (
                "No sufficiently relevant case was found in the available LawWise knowledge base "
                "for your query.\n\n"
                "**To improve case research:**\n"
                "- Upload relevant court judgments to the knowledge base\n"
                "- Try different search terms\n"
                "- For authoritative case research, consult platforms like SCC Online, "
                "Manupatra, or Indian Kanoon, or engage a qualified lawyer"
            ),
            "cases": [],
            "sources": [],
            "agent": "case_research",
            "disclaimer": get_disclaimer(),
            "has_cases": False,
        }

    user_prompt = f"""Research query: {query}

Jurisdiction: {jurisdiction}

RETRIEVED SOURCES FROM LAWWISE KNOWLEDGE BASE:
{context}

Based ONLY on the above sources:
1. Identify any relevant cases or legal principles
2. For each case found IN THE SOURCES, provide:
   - Case name (exactly as stated in source)
   - Court
   - Year  
   - Legal issue
   - Relevant finding
   - Why it may be relevant

If the sources do not contain specific cases but contain relevant legal principles, explain those principles instead and note the absence of specific case citations.

IMPORTANT: Only reference what is in the sources above. Never invent case details."""

    answer = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=2500)

    return {
        "answer": answer,
        "sources": citations,
        "agent": "case_research",
        "disclaimer": get_disclaimer(),
        "has_cases": True,
        "query": query,
    }
