"""Contracts and Qdrant storage for BGE-M3 hybrid retrieval."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol


@dataclass(frozen=True)
class HybridEmbedding:
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    colbert: List[List[float]]


@dataclass(frozen=True)
class HybridChunk:
    id: str
    protocol: str
    page: int
    text: str
    embedding: HybridEmbedding


class HybridEncoder(Protocol):
    def encode(self, texts: Iterable[str]) -> List[HybridEmbedding]: ...


class HybridVectorStore(Protocol):
    def upsert(self, chunks: Iterable[HybridChunk]) -> None: ...

    def search(self, query: HybridEmbedding, limit: int = 8) -> List[HybridChunk]: ...


class QdrantHybridStore:
    """Qdrant adapter using dense, sparse, and late-interaction vectors."""

    def __init__(self, url: str, collection: str, dense_size: int = 1024) -> None:
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.ensure_collection(dense_size)

    def ensure_collection(self, dense_size: int = 1024) -> None:
        models = self.models
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
                "colbert": models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False)),
            },
        )

    def upsert(self, chunks: Iterable[HybridChunk]) -> None:
        points = []
        for chunk in chunks:
            points.append(
                self.models.PointStruct(
                    id=chunk.id,
                    vector={
                        "dense": chunk.embedding.dense,
                        "sparse": self.models.SparseVector(indices=chunk.embedding.sparse_indices, values=chunk.embedding.sparse_values),
                        "colbert": chunk.embedding.colbert,
                    },
                    payload={"protocol": chunk.protocol, "page": chunk.page, "text": chunk.text},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query: HybridEmbedding, limit: int = 8) -> List[Dict[str, Any]]:
        """Return fused candidates; query fusion is kept server-side by Qdrant."""
        models = self.models
        prefetch = [
            models.Prefetch(query=query.dense, using="dense", limit=limit * 2),
            models.Prefetch(query=models.SparseVector(indices=query.sparse_indices, values=query.sparse_values), using="sparse", limit=limit * 2),
            models.Prefetch(query=query.colbert, using="colbert", limit=limit * 2),
        ]
        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": str(point.id), "score": point.score, **(point.payload or {})}
            for point in results.points
        ]
