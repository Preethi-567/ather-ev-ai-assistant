# scripts/test_gemini.py
"""
Manual smoke test: does the Gemini key work, and with which model?

Run:  python scripts/test_gemini.py

This is a script, not a unit test (it spends one real API call), so it does all
its work under `if __name__ == "__main__"` and pytest never runs it.
The model comes from the app's own chain, so this tests what the app will use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from src.generation.gemini_client import MODEL_FALLBACK_CHAIN, call_gemini

    print("Model chain:", MODEL_FALLBACK_CHAIN)
    reply = call_gemini("Say hello in one sentence.", temperature=0.0, max_output_tokens=40)
    if reply:
        print("✅ Gemini working:", reply)
    else:
        print("❌ No reply — see the log lines above (key missing, quota, or model not found).")


if __name__ == "__main__":
    main()
