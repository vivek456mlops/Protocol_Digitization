"""Provider-neutral LLM and embedding boundaries.

Production deployments can point these at any OpenAI-compatible service. Local
mode remains deterministic and does not require credentials.
"""

import os
import hashlib
from typing import Iterable, List, Optional, Protocol

import httpx

from .rag import Chunk, mock_answer


class EmbeddingProvider(Protocol):
    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...


class AnswerProvider(Protocol):
    def answer(self, question: str, sources: list[Chunk], domain_context: Optional[List[str]] = None) -> str: ...


class LocalEmbeddingProvider:
    """Deterministic local feature vectors used until a real model is configured."""

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 128
            for term in text.lower().split():
                index = int(hashlib.sha256(term.encode("utf-8")).hexdigest(), 16) % len(vector)
                vector[index] += 1.0
            vectors.append(vector)
        return vectors


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=60,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


class HuggingFaceEmbeddingProvider:
    def __init__(self, token: str, model: str, base_url: str) -> None:
        self.token = token
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            response = httpx.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.token}"},
                json={"inputs": text},
                timeout=60,
            )
            response.raise_for_status()
            vector = response.json()
            while vector and isinstance(vector[0], list):
                vector = vector[0]
            vectors.append(vector)
        return vectors


class MockAnswerProvider:
    def answer(self, question: str, sources: list[Chunk], domain_context: Optional[List[str]] = None) -> str:
        return mock_answer(question, sources)


class OpenAICompatibleAnswerProvider:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @staticmethod
    def _system_instruction() -> str:
        return "For greetings and simple general conversation, respond naturally. For protocol-specific questions, distinguish the validated protocol excerpts from general domain context. Use protocol excerpts for what this protocol instructs. Use general context to explain common scientific rationale, but never present it as an approved protocol change. For increase, optimization, or deviation questions, you may calculate the proposed new value, but do not say the change might be beneficial, necessary, recommended, or safe. State clearly that the effect is unknown from the provided evidence and that the change requires validation or manufacturer guidance. You may perform simple arithmetic or unit conversions when the source gives the original quantity. Say evidence is insufficient only when neither source type supports the response. Keep citations and label general context clearly."

    def answer(self, question: str, sources: list[Chunk], domain_context: Optional[List[str]] = None) -> str:
        source_text = "\n".join(f"[{source.protocol}, page {source.page}] {source.text}" for source in sources)
        domain_text = "\n".join(f"[GENERAL DOMAIN CONTEXT] {note}" for note in (domain_context or []))
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": self._system_instruction()},
                    {"role": "user", "content": f"Question: {question}\n\nValidated protocol excerpts:\n{source_text}\n\nGeneral domain context (not protocol instructions):\n{domain_text}"},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def configured_answer_provider() -> AnswerProvider:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("HF_API_TOKEN") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAICompatibleAnswerProvider(
            api_key,
            os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
    return MockAnswerProvider()


def configured_embedding_provider() -> EmbeddingProvider:
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "")
    hf_token = os.getenv("HF_API_TOKEN") or (os.getenv("EMBEDDING_API_KEY") if "huggingface.co" in embedding_base_url else None) or (os.getenv("LLM_API_KEY") if "huggingface.co" in llm_base_url else None)
    if hf_token:
        model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        base_url = os.getenv("HF_EMBEDDING_URL", f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction")
        return HuggingFaceEmbeddingProvider(hf_token, model, base_url)
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAICompatibleEmbeddingProvider(
            api_key,
            os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    return LocalEmbeddingProvider()
