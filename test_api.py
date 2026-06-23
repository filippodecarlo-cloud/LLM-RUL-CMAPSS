"""Test rapido: lista modelli disponibili e prova una chiamata."""
import requests
from config import GEMINI_API_KEY

BASE = "https://generativelanguage.googleapis.com"

# 1. Lista modelli disponibili
print("=== Modelli disponibili ===")
r = requests.get(f"{BASE}/v1beta/models?key={GEMINI_API_KEY}", timeout=30)
if r.status_code == 200:
    models = r.json().get("models", [])
    for m in models:
        name = m.get("name", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            print(f"  {name}")
else:
    print(f"  Errore: {r.status_code} - {r.json()}")

# 2. Prova con gemini-2.0-flash via v1
print("\n=== Test chiamata con gemini-2.0-flash (v1) ===")
url = f"{BASE}/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
payload = {"contents": [{"parts": [{"text": "Reply with just the number 42."}]}]}
r = requests.post(url, json=payload, timeout=30)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    print(f"Risposta: {text}")
else:
    print(r.json().get("error", {}).get("message", r.text)[:200])
