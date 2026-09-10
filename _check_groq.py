import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY","")
model   = os.environ.get("GROQ_MODEL","")
print(f"Key prefix : {api_key[:10]}...")
print(f"Model      : {model}")

# Try to list available models
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    models = client.models.list()
    names = sorted([m.id for m in models.data])
    print(f"\nAvailable Groq models ({len(names)}):")
    for n in names:
        print(f"  {n}")
except Exception as e:
    print(f"\nError: {e}")
