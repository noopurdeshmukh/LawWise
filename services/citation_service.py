"""
Citation Service — format and validate legal source citations
"""
from typing import List, Dict, Any


DISCLAIMER = (
    "LawWise provides general legal information for educational purposes only. "
    "It is not a substitute for advice from a qualified lawyer. "
    "Always consult a licensed legal professional for your specific situation."
)


def build_citation(meta: Dict[str, Any], excerpt: str = "", score: float = 0.0) -> Dict[str, Any]:
    return {
        "document": meta.get("source", "Unknown Source"),
        "section": meta.get("section", ""),
        "page": meta.get("page", ""),
        "source_type": meta.get("doc_type", "Legal Document"),
        "jurisdiction": meta.get("jurisdiction", "India"),
        "year": meta.get("year", ""),
        "court": meta.get("court", ""),
        "relevance_score": round(score, 3),
        "excerpt": excerpt[:300] + ("..." if len(excerpt) > 300 else ""),
    }


def format_citations_text(citations: List[Dict[str, Any]]) -> str:
    """Format citations as plain text for LLM prompts."""
    if not citations:
        return "No sources available."
    lines = []
    for i, c in enumerate(citations, 1):
        line = f"[{i}] {c['document']}"
        if c.get("section"):
            line += f" — {c['section']}"
        if c.get("page"):
            line += f" (Page {c['page']})"
        if c.get("year"):
            line += f" ({c['year']})"
        lines.append(line)
    return "\n".join(lines)


def get_disclaimer() -> str:
    return DISCLAIMER
