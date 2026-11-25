from collections import Counter
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from .config import CFG


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int | None
    similarity: float      # vector similarity (cosine)
    score: float = 0.0     # relevance score assigned by the grader


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=CFG.chroma_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=CFG.embedding_model
        )
        self._col = self._client.get_or_create_collection(
            CFG.collection,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks) -> int:
        if not chunks:
            return 0
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self._col.count()

    def query(self, text: str, k: int) -> list[Hit]:
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_texts=[text], n_results=min(k, n))
        hits = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(Hit(id=id_, text=doc, source=meta["source"],
                            page=meta.get("page") or None, similarity=1 - dist))
        return hits

    def list_sources(self) -> dict[str, int]:
        metas = self._col.get(include=["metadatas"])["metadatas"]
        return dict(Counter(m["source"] for m in metas))

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})