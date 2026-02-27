"""
Retrieval engine: queries text and image collections,
normalizes scores, and returns structured results.
"""

# SQLite fix for cluster environments
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb

from mmrag.config import CHROMA_DIR, TEXT_COLLECTION, IMAGE_COLLECTION, TOP_K
from mmrag.schema import RetrievalResult
from mmrag.embeddings import TextEncoder, ImageEncoder


class Retriever:
    def __init__(self):
        print("[Retriever] Initializing...")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.text_col = self.client.get_collection(TEXT_COLLECTION)
        self.image_col = self.client.get_collection(IMAGE_COLLECTION)
        print(f"[Retriever] Text collection:  {self.text_col.count()} docs")
        print(f"[Retriever] Image collection: {self.image_col.count()} docs")

        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

    def retrieve(self, query: str, top_k: int = TOP_K) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """
        Retrieve top-k text and top-k image results for a query.
        Returns (text_results, image_results).
        """
        # Text retrieval: query encoded with MiniLM
        text_emb = self.text_encoder.encode_query(query).tolist()
        text_raw = self.text_col.query(
            query_embeddings=[text_emb],
            n_results=top_k,
            include=["distances", "metadatas"],
        )

        # Image retrieval: query encoded with CLIP text encoder
        image_emb = self.image_encoder.encode_query(query).tolist()
        image_raw = self.image_col.query(
            query_embeddings=[image_emb],
            n_results=top_k,
            include=["distances", "metadatas"],
        )

        text_results = self._parse_results(text_raw, doc_type="text")
        image_results = self._parse_results(image_raw, doc_type="image")

        return text_results, image_results

    def _parse_results(self, raw: dict, doc_type: str) -> list[RetrievalResult]:
        """Convert ChromaDB raw output to RetrievalResult list."""
        results = []
        ids = raw["ids"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        for doc_id, dist, meta in zip(ids, distances, metadatas):
            # ChromaDB cosine distance is in [0, 2]. Convert to similarity in [-1, 1].
            score = 1.0 - dist

            results.append(RetrievalResult(
                doc_id=doc_id,
                score=round(score, 4),
                doc_type=doc_type,
                content=meta.get("content", ""),
                image_path=meta.get("image_path", ""),
                group_id=meta.get("group_id", ""),
                metadata=meta,
            ))

        return results


def print_results(text_results: list[RetrievalResult], image_results: list[RetrievalResult]):
    """Print retrieval results to stdout (required before generation)."""
    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\n--- Text Results (top {len(text_results)}) ---")
    for i, r in enumerate(text_results):
        print(f"  [{i+1}] ID: {r.doc_id}")
        print(f"      Score: {r.score}")
        print(f"      Group: {r.group_id}")
        print(f"      Content: {r.content[:120]}...")
        print()

    print(f"--- Image Results (top {len(image_results)}) ---")
    for i, r in enumerate(image_results):
        print(f"  [{i+1}] ID: {r.doc_id}")
        print(f"      Score: {r.score}")
        print(f"      Group: {r.group_id}")
        print(f"      Path: {r.image_path}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = Retriever()

    query = "What does the diagram show about the life cycle?"
    print(f"\nQuery: {query}\n")

    text_results, image_results = retriever.retrieve(query, top_k=3)
    print_results(text_results, image_results)