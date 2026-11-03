import os
import time
import requests

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class NVIDIAHostedEmbeddings(Embeddings):

    def __init__(self, model=None, batch_size=1):

        self.api_key = os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing")

        self.model = (
            model
            or os.getenv(
                "NVIDIA_EMBED_MODEL",
                "nvidia/nemotron-3.5-embed-8b"
            )
        )

        # Keep this at 1 while troubleshooting
        self.batch_size = batch_size

        self.url = "https://integrate.api.nvidia.com/v1/embeddings"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _request_embeddings(self, texts, input_type):

        payload = {
            "model": self.model,
            "input": texts,
            "input_type": input_type,
            "modality": "text",
            "encoding_format": "float",
            "truncate": "END"
        }

        for attempt in range(1, 4):

            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )

                if response.ok:
                    result = response.json()

                    data = sorted(
                        result["data"],
                        key=lambda item: item["index"]
                    )

                    return [
                        item["embedding"]
                        for item in data
                    ]

                print(
                    f"NVIDIA API attempt {attempt} failed: "
                    f"HTTP {response.status_code}"
                )

                if response.status_code < 500:
                    raise RuntimeError(
                        f"NVIDIA API error: "
                        f"{response.status_code} - {response.text}"
                    )

            except requests.RequestException as error:
                print(f"Network error on attempt {attempt}: {error}")

            if attempt < 3:
                time.sleep(attempt * 2)

        raise RuntimeError(
            f"NVIDIA embedding failed after 3 attempts. "
            f"Input count: {len(texts)}, "
            f"Input length(s): {[len(t) for t in texts]}"
        )

    def _embed(self, texts, input_type):

        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):

            batch = texts[start:start + self.batch_size]

            print(
                f"Embedding batch: {start + 1} "
                f"to {start + len(batch)} "
                f"of {len(texts)}"
            )

            embeddings = self._request_embeddings(
                batch,
                input_type
            )

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_documents(self, texts):
        return self._embed(texts, input_type="passage")

    def embed_query(self, text):
        return self._embed([text], input_type="query")[0]