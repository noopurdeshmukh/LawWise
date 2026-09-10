import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import route_to_agent

result = route_to_agent("What does indemnity mean in a contract?", jurisdiction="India")

answer = result.get("answer") or result.get("raw_analysis","")
# encode-safe print
safe = answer.encode('ascii', errors='replace').decode('ascii')
print("ANSWER OK, length:", len(answer))
print("First 300 chars:", safe[:300])
print("Agent:", result.get("primary_agent"))
print("Sources:", len(result.get("sources",[])))
for s in result.get("sources",[]):
    doc = s.get("document","?")
    score = s.get("relevance_score", 0)
    print(f"  [{score:.3f}] {doc[:60]}")
