"""Full end-to-end test of the Ask LawWise flow."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

print("Model:", os.environ.get("GROQ_MODEL"))
print("Key  :", os.environ.get("GROQ_API_KEY","")[:12] + "...")

# Test orchestrator -> agent -> LLM
from agents.orchestrator import route_to_agent

result = route_to_agent(
    "What does indemnity mean in a contract?",
    jurisdiction="India"
)

print("\n--- Answer (first 400 chars) ---")
print((result.get("answer") or result.get("raw_analysis",""))[:400])
print("\n--- Agent used:", result.get("primary_agent"))
print("--- Sources   :", len(result.get("sources",[])))
print("--- Steps     :", [s["step"] for s in result.get("agent_steps",[])])
