"""Small retrieval core used by the API and tests.

The interfaces are intentionally provider-neutral: replace the in-memory store and
answer generator with a vector store and LLM without changing the HTTP contract.
"""

from dataclasses import dataclass
import re
from typing import Iterable, Protocol

STOPWORDS = {"a", "an", "and", "are", "is", "of", "on", "the", "to", "what", "with"}


class EmbeddingProvider(Protocol):
    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...

@dataclass(frozen=True)
class Chunk:
    protocol: str
    page: int
    text: str


class InMemoryRetriever:
    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self.chunks = list(chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, limit: int = 4) -> list[Chunk]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
        scored = [
            (len(terms.intersection(set(re.findall(r"[a-z0-9]+", chunk.text.lower())))), chunk)
            for chunk in self.chunks
        ]
        ranked = [chunk for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
        return ranked[:limit]


class EmbeddingRetriever:
    def __init__(self, chunks: Iterable[Chunk], provider: EmbeddingProvider, minimum_score: float = 0.55) -> None:
        self.provider = provider
        self.minimum_score = minimum_score
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.replace(chunks)

    def replace(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        self.vectors = self.provider.embed(chunk.text for chunk in self.chunks)

    def search(self, query: str, limit: int = 4) -> list[Chunk]:
        query_vector = self.provider.embed([query])[0]
        scored = [(self._cosine(query_vector, vector), chunk) for chunk, vector in zip(self.chunks, self.vectors)]
        return [chunk for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True) if score >= self.minimum_score][:limit]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        size = min(len(left), len(right))
        numerator = sum(left[index] * right[index] for index in range(size))
        left_norm = sum(value * value for value in left[:size]) ** 0.5
        right_norm = sum(value * value for value in right[:size]) ** 0.5
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def mock_answer(query: str, sources: list[Chunk]) -> str:
    if not sources:
        return "I could not find support for that question in the indexed protocols."
    excerpts = " ".join(source.text for source in sources[:2])
    return f"Based on the indexed protocol: {excerpts}"
