"""
Vector Store — FAISS-based storage and persistence for email chunk embeddings.
"""

import os
import json
import numpy as np
import faiss

FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_METADATA_PATH = "data/chunks_metadata.json"


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[dict] = []  # parallel list of chunk dicts

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """Add embeddings and their corresponding chunks to the store."""
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """
        Search for the top-k most similar chunks.

        Returns:
            List of chunk dicts with added 'score' field.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["score"] = float(dist)
                results.append(result)

        return results

    def save(self, index_path: str = FAISS_INDEX_PATH, meta_path: str = CHUNKS_METADATA_PATH):
        """Persist index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved FAISS index ({self.index.ntotal} vectors) and metadata.")

    def load(self, index_path: str = FAISS_INDEX_PATH, meta_path: str = CHUNKS_METADATA_PATH) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        self.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"Loaded FAISS index ({self.index.ntotal} vectors) and {len(self.chunks)} chunks.")
        return True
