# scripts/check_setup.py
"""
Run this script once to verify every dependency is correctly installed.
If all checks pass, you're ready to build Phase 2.
"""

import sys

checks = []

def check(name, fn):
    try:
        fn()
        checks.append((name, "✅ OK"))
    except Exception as e:
        checks.append((name, f"❌ FAILED: {e}"))

# Check each dependency
check("python-dotenv",     lambda: __import__("dotenv"))
check("google-genai", lambda: __import__("google.genai"))
check("sentence-transformers", lambda: __import__("sentence_transformers"))
check("chromadb",          lambda: __import__("chromadb"))
check("rank-bm25",         lambda: __import__("rank_bm25"))
check("PyMuPDF (fitz)",    lambda: __import__("fitz"))
check("fastapi",           lambda: __import__("fastapi"))
check("streamlit",         lambda: __import__("streamlit"))
check("pyttsx3",           lambda: __import__("pyttsx3"))
check("openai-whisper",    lambda: __import__("whisper"))
check("ragas",             lambda: __import__("ragas"))
check("langgraph",         lambda: __import__("langgraph"))
check("loguru",            lambda: __import__("loguru"))
check("pillow",            lambda: __import__("PIL"))
check("sounddevice",       lambda: __import__("sounddevice"))
check("scipy",             lambda: __import__("scipy"))
check("torch",             lambda: __import__("torch"))

# Print results
print("\n── Ather EV Assistant — Setup Check ──")
all_ok = True
for name, result in checks:
    print(f"  {result}  {name}")
    if "FAILED" in result:
        all_ok = False

print()
if all_ok:
    print("🎉 All dependencies OK. Ready for Phase 2.")
else:
    print("⚠️  Fix the failed items above before continuing.")
    sys.exit(1)