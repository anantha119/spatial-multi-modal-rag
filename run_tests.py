"""
Batch runner for the 6 required test queries.
Outputs to both stdout and logs/test_results.log.
"""

import sys
from pathlib import Path
from datetime import datetime

from mmrag.config import LOGS_DIR
from mmrag.pipeline import RAGPipeline


# The 6 required test queries from the assignment
TEST_QUERIES = [
    # Diagram-Oriented (1-3)
    "What does the diagram question say about the correct choice? Return the answer and cite the diagram.",
    "Find a relevant AI2D diagram about a process or cycle. Summarize what it shows in 2 sentences. Cite.",
    "Retrieve a diagram and its paired text with the same group_id. Explain how you linked them. Cite both.",
    # Chart-Oriented (4-6)
    "Find a chart about comparison across categories. State which category is highest. Cite the chart.",
    "Find a chart question that requires reading values. Answer it and cite chart + text evidence.",
    "Retrieve a chart and produce a one-sentence operator summary (what a technician would do next). Cite.",
]


class TeeOutput:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Tee output to both terminal and log file
    tee = TeeOutput(str(log_path))
    sys.stdout = tee

    print(f"Multimodal RAG Test Run")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Log file:  {log_path}")
    print(f"Queries:   {len(TEST_QUERIES)}")
    print()

    pipeline = RAGPipeline()
    results = pipeline.run_batch(TEST_QUERIES)

    print(f"\nLog saved to: {log_path}")

    # Restore stdout and close log
    sys.stdout = tee.terminal
    tee.close()

    print(f"\nDone. Log saved to: {log_path}")


if __name__ == "__main__":
    main()