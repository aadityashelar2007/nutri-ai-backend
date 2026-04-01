import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"✅ Key Found: {key[:5]}...{key[-5:]}")
    if "X" in key:
        print("❌ ERROR: Your key still has 'X' characters in it!")
else:
    print("❌ ERROR: Python cannot see your .env file at all.")