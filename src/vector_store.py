import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str, index_dir: Path):
        self.model = SentenceTransformer(model_name)
        self.index_dir = index_dir
        self.index_path = index_dir / "docs.faiss"
        self.meta_path = index_dir / "meta.json"
        self.index = None
        self.metadata: List[Dict[str, str]] = []

    def build(self, documents: List[Dict[str, str]]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = documents
        embeddings = self.model.encode(
            [doc["content"] for doc in documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return True

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        if self.index is None:
            return []
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, indices = self.index.search(query_vector, top_k)

        results: List[Dict[str, str]] = []
        for idx in indices[0]:
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append(self.metadata[idx])
        return results
