import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Step 1: Check available models
print("\nChecking NVIDIA available models...")

response = requests.get(
    f"{BASE_URL}/models",
    headers=headers,
    timeout=30
)

print("Models API status:", response.status_code)

if not response.ok:
    print(response.text)
    raise SystemExit()

models = response.json().get("data", [])

print("\nAvailable model IDs:")
for model in models:
    print(model.get("id"))

# Step 2: Test chat completion
# Replace this with a model ID from the list above.
MODEL = "openai/gpt-oss-20b"

if not MODEL:
    raise ValueError(
        "Set NVIDIA_CHAT_MODEL in your .env using a model ID from the list."
    )

print(f"\nTesting chat model: {MODEL}")

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Explain React in two sentences."
        }
    ],
    "temperature": 0.2,
    "max_tokens": 200,
    "stream": False
}

response = requests.post(
    f"{BASE_URL}/chat/completions",
    headers=headers,
    json=payload,
    timeout=120
)

print("\nChat API status:", response.status_code)

if response.ok:
    print("\nChat response:")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("\nChat API error:")
    print(response.text)