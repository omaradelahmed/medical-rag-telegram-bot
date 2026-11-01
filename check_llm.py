"""Step 6a: check that the API key in .env works, list the available models, test one call.
Run:  uv run check_llm.py
The output NEVER prints your key, so it is safe to paste it back to me.
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
model = os.getenv("LLM_MODEL")

if not key:
    sys.exit("LLM_API_KEY not found. Check that .env is in this folder and the line is: LLM_API_KEY=your_key")

problems = []
if key != key.strip() or key[0] in "\"'" or key[-1] in "\"'":
    problems.append("the key has quotes or spaces around it: remove them in .env")
print(f"Key loaded: {len(key)} characters" + (f"  (WARNING: {problems[0]})" if problems else ""))
print(f"Base URL: {base_url}")

client = OpenAI(base_url=base_url, api_key=key.strip().strip("\"'"))

try:
    models = sorted(m.id for m in client.models.list().data)
except Exception as e:  # wrong key, wrong URL, no internet...
    sys.exit(f"Could not list models: {type(e).__name__}: {e}")

print(f"\nAvailable models ({len(models)}):")
for m in models:
    print("  -", m)

if not model:
    print("\nNext: pick a model from the list and add this line to .env:")
    print("  LLM_MODEL=model-name-here")
    sys.exit(0)

print(f"\nTesting LLM_MODEL={model} ...")
t0 = time.perf_counter()
try:
    extra = {"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in model else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=500,  # reasoning models spend tokens on thinking: 10 was too small
        temperature=0,
        **extra,
    )
    choice = resp.choices[0]
    print(f"Reply: {choice.message.content!r}  finish_reason={choice.finish_reason}  "
          f"({time.perf_counter() - t0:.1f}s)")
except Exception as e:
    print(f"Chat call failed: {type(e).__name__}: {e}")
