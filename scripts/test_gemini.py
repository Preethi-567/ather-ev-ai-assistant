# scripts/test_gemini.py
import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from typing import TypedDict, Any, List, Dict

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents="Say hello in one sentence.",
        )
        print("✅ Gemini working:", response.text)
        break
    except errors.ClientError as e:
        if "429" in str(e):
            wait = 10 * (attempt + 1)
            print(f"⏳ Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        else:
            raise