"""
End-to-end RAG pipeline: retrieve -> print results -> build context -> generate -> validate.
"""
# SQLite fix for cluster environments
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from mmrag.retrieval import Retriever, print_results
from mmrag.context import ContextBuilder
from mmrag.generate import generate_answer, print_generation
from mmrag.config import TOP_K, SCORE_THRESHOLD

INSUFFICIENT_EVIDENCE = "The retrieved evidence is insufficient to answer this query."


class RAGPipeline:
    def __init__(self):
        print("[Pipeline] Initializing components...")
        self.retriever = Retriever()
        self.context_builder = ContextBuilder()
        print("[Pipeline] Ready.\n")

    def _check_score_threshold(self, text_results, image_results) -> bool:
        """
        Check if retrieval scores meet minimum confidence.
        Returns True if evidence is sufficient, False otherwise.
        Only checks text scores since text carries the semantic content
        the LLM uses for generation. Image scores from CLIP operate in
        a different range (~0.25-0.35) and are not directly comparable.
        """
        if not text_results:
            return False
        best_text_score = text_results[0].score
        if best_text_score < SCORE_THRESHOLD:
            print(f"\n[Pipeline] Best text score ({best_text_score}) below threshold ({SCORE_THRESHOLD}). Insufficient evidence.")
            return False
        return True

    def _insufficient_evidence_result(self) -> dict:
        """Return a standardized insufficient-evidence response."""
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "validated_answer": INSUFFICIENT_EVIDENCE,
            "citations_found": [],
            "valid_citations": [],
            "invalid_citations": [],
        }

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Run the full pipeline for a single query.
        Returns the generation result dict with citation validation.
        """
        print(f"\nQUERY: {query}")
        print(f"TOP_K: {top_k}")

        # Step 1: Retrieve
        text_results, image_results = self.retriever.retrieve(query, top_k=top_k)
        print_results(text_results, image_results)

        # Step 2: Score threshold check
        if not self._check_score_threshold(text_results, image_results):
            result = self._insufficient_evidence_result()
            print_generation(query, result)
            return result

        # Step 3: Build context + collect image paths for VLM
        context, valid_ids, image_paths = self.context_builder.build(query, text_results, image_results)

        # Step 4: Generate with VLM (images + text) and validate
        result = generate_answer(query, context, valid_ids, image_paths)
        print_generation(query, result)

        return result

    def run_batch(self, queries: list[str], top_k: int = TOP_K) -> list[dict]:
        """Run the pipeline on a list of queries."""
        results = []
        for i, query in enumerate(queries):
            print("\n" + "#" * 70)
            print(f"  QUERY {i+1} of {len(queries)}")
            print("#" * 70)
            result = self.run(query, top_k=top_k)
            result["query"] = query
            results.append(result)

        # Print summary
        print("\n" + "#" * 70)
        print("  BATCH SUMMARY")
        print("#" * 70)
        for i, r in enumerate(results):
            total = len(r["citations_found"])
            valid = len(r["valid_citations"])
            invalid = len(r["invalid_citations"])

            if INSUFFICIENT_EVIDENCE in r.get("validated_answer", ""):
                status = "ABSTAIN"
            elif invalid > 0:
                status = "FAIL"
            elif total > 0:
                status = "PASS"
            else:
                status = "WARN"

            print(f"  [{status}] Q{i+1}: {total} citations, {valid} valid, {invalid} invalid")
            print(f"         {r['query'][:80]}")
        print("#" * 70)

        return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run("What does the diagram show about the life cycle?")