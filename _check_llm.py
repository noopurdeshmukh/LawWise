import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

# Override model to one that is available
os.environ["GROQ_MODEL"] = "openai/gpt-oss-20b"

from services.groq_service import call_llm
result = call_llm(
    "You are a helpful legal assistant.",
    "In one sentence, what is indemnity in a contract?",
    temperature=0.1,
    max_tokens=100
)
print("LLM response:", result)
