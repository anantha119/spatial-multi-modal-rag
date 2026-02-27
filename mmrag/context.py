"""
Context assembly: builds the structured evidence block for the LLM
from retrieval results, including bidirectional group_id-based linkage
between text and image documents.
"""
from mmrag.schema import RetrievalResult, Document, load_documents
from mmrag.config import DOCS_DIR


class ContextBuilder:
    def __init__(self):
        """Load all documents to enable bidirectional group_id lookups."""
        print("[ContextBuilder] Loading documents for group_id linkage...")

        text_docs = load_documents(str(DOCS_DIR / "text_documents.json"))
        image_docs = load_documents(str(DOCS_DIR / "image_documents.json"))

        # Build group_id -> list of text content for fast lookup
        self.group_texts: dict[str, list[dict]] = {}
        for d in text_docs:
            if d.group_id not in self.group_texts:
                self.group_texts[d.group_id] = []
            self.group_texts[d.group_id].append({
                "doc_id": d.doc_id,
                "content": d.content,
            })

        # Build group_id -> image doc_id and path for reverse lookup
        self.group_images: dict[str, dict] = {}
        for d in image_docs:
            self.group_images[d.group_id] = {
                "doc_id": d.doc_id,
                "image_path": d.image_path,
            }

        print(f"[ContextBuilder] Ready. {len(self.group_texts)} text groups, {len(self.group_images)} image groups indexed.")

    def build(
        self,
        query: str,
        text_results: list[RetrievalResult],
        image_results: list[RetrievalResult],
    ) -> tuple[str, set[str], list[tuple[str, str]]]:
        """
        Build the context block for the LLM prompt.
        Returns (context_string, valid_ids, image_paths) where:
            - valid_ids: set of all document IDs the LLM is allowed to cite
            - image_paths: list of (doc_id, file_path) tuples for the VLM
        """
        valid_ids: set[str] = set()
        image_paths: list[tuple[str, str]] = []
        seen_images: set[str] = set()  # deduplicate images by doc_id
        lines = ["Retrieved Evidence:", ""]

        # Text results with linked images from same group_id
        lines.append("TEXT RESULTS:")
        for r in text_results:
            valid_ids.add(r.doc_id)
            lines.append(
                f"[{r.doc_id}] (score: {r.score}, group: {r.group_id}) "
                f"{r.content}"
            )

            # Reverse lookup: find paired image via group_id
            linked_img = self.group_images.get(r.group_id)
            if linked_img:
                valid_ids.add(linked_img["doc_id"])
                lines.append(
                    f"  Paired image: [{linked_img['doc_id']}] "
                    f"Path: {linked_img['image_path']}"
                )
                # Collect image for VLM
                if linked_img["doc_id"] not in seen_images:
                    image_paths.append((linked_img["doc_id"], linked_img["image_path"]))
                    seen_images.add(linked_img["doc_id"])

        lines.append("")

        # Image results with linked text from same group_id
        lines.append("IMAGE RESULTS:")
        for r in image_results:
            valid_ids.add(r.doc_id)
            lines.append(
                f"[{r.doc_id}] (score: {r.score}, group: {r.group_id}) "
                f"Image path: {r.image_path}"
            )

            # Collect image for VLM
            if r.doc_id not in seen_images and r.image_path:
                image_paths.append((r.doc_id, r.image_path))
                seen_images.add(r.doc_id)

            # Forward lookup: find paired text records via group_id
            linked = self.group_texts.get(r.group_id, [])
            if linked:
                lines.append(f"  Linked text from group {r.group_id}:")
                for lt in linked[:3]:  # Cap at 3 linked texts to avoid bloat
                    valid_ids.add(lt["doc_id"])
                    lines.append(f"    [{lt['doc_id']}] {lt['content']}")
            else:
                lines.append(f"  No linked text found for group {r.group_id}")

            lines.append("")

        context = "\n".join(lines)
        return context, valid_ids, image_paths


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # SQLite fix
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    from mmrag.retrieval import Retriever, print_results

    retriever = Retriever()
    builder = ContextBuilder()

    query = "Find a relevant AI2D diagram about a process or cycle."
    text_results, image_results = retriever.retrieve(query, top_k=3)
    print_results(text_results, image_results)

    context, valid_ids, image_paths = builder.build(query, text_results, image_results)
    print("\n--- CONTEXT BLOCK ---")
    print(context)
    print(f"\n--- VALID CITATION IDS ({len(valid_ids)}) ---")
    for vid in sorted(valid_ids):
        print(f"  {vid}")
    print(f"\n--- IMAGES FOR VLM ({len(image_paths)}) ---")
    for doc_id, path in image_paths:
        print(f"  {doc_id}: {path}")