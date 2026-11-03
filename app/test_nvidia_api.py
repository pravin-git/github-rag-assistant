
import os
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

URL = "https://integrate.api.nvidia.com/v1/embeddings"

payload = {
    "model": "nvidia/nemotron-3.5-embed-8b",
    "input": [
        "React is a JavaScript library for building user interfaces."
    ],
    "input_type": "passage",
    "modality": "text",
    "encoding_format": "float",
    "truncate": "END"
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

response = requests.post(
    URL,
    headers=headers,
    json=payload,
    timeout=120
)

print("HTTP Status:", response.status_code)
print("Response:")

try:
    result = response.json()
    print(result)

    if response.ok:
        embedding = result["data"][0]["embedding"]
        print("\nEmbedding successful!")
        print("Dimensions:", len(embedding))

except ValueError:
    print(response.text)
