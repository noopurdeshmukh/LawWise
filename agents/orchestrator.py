"""
LawWise AI Orchestrator — routes user requests to appropriate specialized agents
"""
import logging
import re
from typing import Dict, Any, List, Tuple

from services.groq_service import call_llm
from services.citation_service import get_disclaimer

logger = logging.getLogger(__name__)

# Agent intent keywords
INTENT_PATTERNS = {
    "contract_review": [
        "review contract", "analyze contract", "check contract", "contract analysis",
        "rental agreement", "employment agreement", "service agreement", "nda",
        "non-disclosure", "memorandum", "mou", "review this agreement",
        "risky clause", "problematic clause", "unfair clause", "contract terms",
    ],
    "compliance": [
        "compliance", "comply", "compliant", "regulation", "regulatory",
        "gdpr", "pdp", "privacy policy", "data protection", "it act",
        "satisfy requirement", "meet requirement", "legal requirement",
        "policy check", "policy review",
    ],
    "case_research": [
        "case law", "judgment", "judgement", "precedent", "court order",
        "supreme court", "high court", "ruling", "find cases", "relevant cases",
        "legal precedent", "case related to", "judgments related",
    ],
    "document_analysis": [
        "analyze document", "analyse document", "explain this document",
        "what does this document mean", "legal notice", "explain notice",
        "explain this agreement", "what does this mean", "analyze this",
        "what is this document", "explain document",
    ],
    "qa_rag": [
        "what is", "what does", "what are", "explain", "define",
        "meaning of", "tell me about", "how does", "difference between",
        "what happens", "is it legal", "what section", "which law",
        "can i", "am i entitled", "my rights", "legal meaning",
        "indemnity", "liability", "contract", "bail", "anticipatory",
        "fir", "ipc", "crpc", "consumer", "tenant", "landlord",
    ],
}


def detect_intent(query: str) -> List[str]:
    """
    Detect the user's intent from the query.
    Returns list of matching agent names, ordered by confidence.
    """
    q_lower = query.lower()
    scores = {}

    for agent, patterns in INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if p in q_lower)
        if score > 0:
            scores[agent] = score

    if not scores:
        return ["qa_rag"]

    # Sort by score descending
    sorted_agents = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
    return sorted_agents


def route_to_agent(
    query: str,
    document_text: str = "",
    filename: str = "",
    jurisdiction: str = "India",
) -> Dict[str, Any]:
    """
    Main orchestration function.
    Detects intent, selects agent(s), and returns structured response.
    """
    intents = detect_intent(query)
    primary_intent = intents[0]

    # Build agent activity steps
    steps = [
        {"step": "Understanding your question", "status": "done"},
        {"step": f"Selecting specialized agent: {_agent_display_name(primary_intent)}", "status": "done"},
    ]

    logger.info(f"Orchestrator routing to: {primary_intent} | Query: {query[:60]}")

    try:
        # If document provided and no clear contract/document intent, detect from content
        if document_text and primary_intent == "qa_rag":
            doc_len = len(document_text.strip())
            if doc_len > 500:
                primary_intent = "document_analysis"
                steps[1]["step"] = f"Selecting specialized agent: {_agent_display_name(primary_intent)}"

        steps.append({"step": "Searching legal knowledge base", "status": "done"})
        steps.append({"step": "Retrieving relevant sources", "status": "done"})
        steps.append({"step": "Analyzing information", "status": "done"})
        steps.append({"step": "Preparing response", "status": "done"})

        result = _call_agent(primary_intent, query, document_text, filename, jurisdiction)

        # Handle multi-agent: if both contract review AND case research requested
        if len(intents) >= 2 and "case_research" in intents and primary_intent == "contract_review":
            steps.append({"step": "Running Case Research Agent for supporting judgments", "status": "done"})
            import agents.case_research as case_research_agent
            case_result = case_research_agent.run(
                f"cases related to {query}", jurisdiction
            )
            result["supplementary_case_research"] = case_result

        result["agent_steps"] = steps
        result["primary_agent"] = primary_intent
        return result

    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return {
            "answer": f"An error occurred while processing your request: {str(e)}\n\nPlease try again or check your API configuration.",
            "agent_steps": steps,
            "primary_agent": primary_intent,
            "error": True,
            "disclaimer": get_disclaimer(),
        }


def _call_agent(
    agent_name: str,
    query: str,
    document_text: str,
    filename: str,
    jurisdiction: str,
) -> Dict[str, Any]:
    """Dispatch to the appropriate specialized agent."""
    if agent_name == "contract_review":
        import agents.contract_reviewer as agent
        if document_text:
            return agent.run(document_text, filename)
        else:
            # No document — use Q&A agent with contract focus
            import agents.qa_rag_agent as qa_agent
            return qa_agent.run(query, jurisdiction=jurisdiction)

    elif agent_name == "compliance":
        import agents.compliance_checker as agent
        return agent.run(document_text, business_description=query, jurisdiction=jurisdiction)

    elif agent_name == "case_research":
        import agents.case_research as agent
        return agent.run(query, jurisdiction)

    elif agent_name == "document_analysis":
        import agents.document_analyzer as agent
        if document_text:
            return agent.run(document_text, filename)
        else:
            import agents.qa_rag_agent as qa_agent
            return qa_agent.run(query, jurisdiction=jurisdiction)

    else:  # qa_rag (default)
        import agents.qa_rag_agent as agent
        return agent.run(query, document_text=document_text[:2000] if document_text else "", jurisdiction=jurisdiction)


def _agent_display_name(agent_name: str) -> str:
    names = {
        "qa_rag": "Q&A Legal Research Agent",
        "contract_review": "Contract Reviewer Agent",
        "compliance": "Compliance Checker Agent",
        "case_research": "Case Research Agent",
        "document_analysis": "Document Analyzer Agent",
    }
    return names.get(agent_name, "Legal Research Agent")
