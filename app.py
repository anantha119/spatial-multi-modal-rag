"""
CLI entrypoint for the multimodal RAG system.

Usage:
    python app.py --query "What does the diagram show about the life cycle?"
    python app.py --query "Find a chart about comparison" --top_k 3
"""

import argparse
from mmrag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG System")
    parser.add_argument("--query", type=str, required=True, help="The query to answer")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results per modality")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.run(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()