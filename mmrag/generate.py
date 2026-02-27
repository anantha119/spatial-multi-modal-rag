"""
Grounded answer generation with citation validation.
Calls Qwen3-VL with retrieved context + images, then validates that all
citations in the response reference real retrieved document IDs.
"""

import re
import torch
from pathlib import Path
from PIL import Image

from mmrag.config import LLM_PROVIDER, LLM_MODEL, MAX_TOKENS, SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# System prompt v1 (conservative: abstains too aggressively on Q1, Q6)
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """You are a retrieval-augmented answering system. Your task is to answer the user's query using ONLY the retrieved evidence provided below. You will be shown retrieved images alongside text evidence.
#
# STRICT RULES:
# 1. Cite document IDs in square brackets, e.g. [ai2d_0042_txt_0]. Cite each ID at most once.
# 2. You may ONLY cite IDs that appear in the Retrieved Evidence section.
# 3. When answering about a chart or diagram, ALWAYS cite both the image ID and relevant text IDs.
# 4. Use the provided images to visually verify claims. Describe what you see in the images when relevant.
# 5. If the retrieved evidence is insufficient to answer the query, explicitly state: "The retrieved evidence is insufficient to answer this query."
# 6. Do not fabricate any information or citations.
# 7. Never reproduce raw scores, group IDs, or metadata from the evidence block in your answer.
# 8. Synthesize information into concise, natural sentences. Do not list evidence items.
# 9. When explaining how text and image are linked, reference the shared group_id prefix.
# 10. If you cannot derive a specific claim from the evidence, do not invent one.
#
# EXAMPLE FORMAT:
# Query: What does the chart show about sales?
# Answer: The chart shows that Product A had the highest sales at $500M. [chartqa_0012_img] [chartqa_0012_txt_0]
# """

# ---------------------------------------------------------------------------
# System prompt v2 (balanced: best-effort answers from partial evidence)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a retrieval-augmented answering system. Your task is to answer the user's query using ONLY the retrieved evidence provided below. You will be shown retrieved images alongside text evidence.

STRICT RULES:
1. Cite document IDs in square brackets, e.g. [ai2d_0042_txt_0]. Cite each ID at most once.
2. You may ONLY cite IDs that appear in the Retrieved Evidence section.
3. When answering about a chart or diagram, ALWAYS cite both the image ID and relevant text IDs.
4. Use the provided images to visually verify claims. Describe what you see in the images when relevant.
5. Always provide a best-effort answer if ANY retrieved result is even partially relevant. Only state "The retrieved evidence is insufficient to answer this query." if NONE of the retrieved results have any relevance whatsoever.
6. Do not fabricate any information or citations.
7. Never reproduce raw scores, group IDs, or metadata from the evidence block in your answer.
8. Synthesize information into concise, natural sentences. Do not list evidence items.
9. When explaining how text and image are linked, reference the shared group_id prefix.
10. If you cannot derive a specific claim from the evidence, do not invent one.
11. When asked for an operator summary or action recommendation, you may make reasonable inferences based on what the chart or diagram visually shows, as long as you cite the source.

EXAMPLE FORMAT:
Query: What does the chart show about sales?
Answer: The chart shows that Product A had the highest sales at $500M. [chartqa_0012_img] [chartqa_0012_txt_0]
"""

# ---------------------------------------------------------------------------
# Local model singleton (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_local_model = None
_local_processor = None


def _get_local_model():
    """Load Qwen3-VL model and processor on first call, cache for reuse."""
    global _local_model, _local_processor
    if _local_model is None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"[Generate] Loading VLM: {LLM_MODEL}...")
        _local_model = Qwen3VLForConditionalGeneration.from_pretrained(
            LLM_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        _local_processor = AutoProcessor.from_pretrained(LLM_MODEL)
        print("[Generate] VLM loaded.")
    return _local_model, _local_processor


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    context: str,
    valid_ids: set[str],
    image_paths: list[str] | None = None,
) -> dict:
    """
    Generate a grounded answer using the VLM.

    Args:
        query: the user's question
        context: assembled text evidence block
        valid_ids: set of doc IDs the LLM is allowed to cite
        image_paths: list of (doc_id, path) tuples for images to show the VLM

    Returns dict with keys:
        - answer: the raw LLM response
        - validated_answer: response with invalid citations flagged
        - citations_found: list of all citation IDs found in response
        - valid_citations: list of citations that match retrieved IDs
        - invalid_citations: list of citations that do NOT match
    """
    if image_paths is None:
        image_paths = []

    raw_answer = _call_llm(query, context, image_paths)

    result = _validate_citations(raw_answer, valid_ids)
    return result


# ---------------------------------------------------------------------------
# LLM dispatch
# ---------------------------------------------------------------------------

def _call_llm(query: str, context: str, image_paths: list[tuple[str, str]]) -> str:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "local":
        return _call_local(query, context, image_paths)
    elif LLM_PROVIDER == "anthropic":
        user_prompt = f"{context}\n\nUser Query: {query}"
        return _call_anthropic(user_prompt)
    elif LLM_PROVIDER == "openai":
        user_prompt = f"{context}\n\nUser Query: {query}"
        return _call_openai(user_prompt)
    elif LLM_PROVIDER == "ollama":
        user_prompt = f"{context}\n\nUser Query: {query}"
        return _call_ollama(user_prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def _call_local(query: str, context: str, image_paths: list[tuple[str, str]]) -> str:
    """Run inference with Qwen3-VL (vision-language model)."""
    model, processor = _get_local_model()

    # Build multimodal user content: images first, then text context + query
    user_content = []

    # Add retrieved images with their doc_id labels
    for doc_id, img_path in image_paths:
        if Path(img_path).exists():
            user_content.append({"type": "text", "text": f"[Image: {doc_id}]"})
            user_content.append({"type": "image", "image": img_path})

    # Add the text evidence and query
    user_content.append({
        "type": "text",
        "text": f"{context}\n\nUser Query: {query}",
    })

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    # Process inputs through the VL processor
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate with greedy decoding for deterministic citations
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        temperature=None,
    )

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()


def _call_anthropic(user_prompt: str) -> str:
    """Call Anthropic API."""
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_openai(user_prompt: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _call_ollama(user_prompt: str) -> str:
    """Call local Ollama instance."""
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "options": {"num_predict": MAX_TOKENS},
        },
    )
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

def _validate_citations(answer: str, valid_ids: set[str]) -> dict:
    """
    Extract all bracketed citations from the answer and check
    against the set of valid retrieved IDs.
    """
    pattern = r'\[((?:ai2d|chartqa)_\d+_(?:txt_\d+|img))\]'
    citations_found = re.findall(pattern, answer)

    valid = [c for c in citations_found if c in valid_ids]
    invalid = [c for c in citations_found if c not in valid_ids]

    validated_answer = answer
    for inv in invalid:
        validated_answer = validated_answer.replace(
            f"[{inv}]", f"[INVALID: {inv}]"
        )

    return {
        "answer": answer,
        "validated_answer": validated_answer,
        "citations_found": citations_found,
        "valid_citations": valid,
        "invalid_citations": invalid,
    }


def print_generation(query: str, result: dict):
    """Print the generation output with citation validation summary."""
    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(f"\nQuery: {query}\n")
    print(result["validated_answer"])
    print("\n" + "-" * 70)
    print("CITATION VALIDATION")
    print("-" * 70)
    print(f"  Total citations found: {len(result['citations_found'])}")
    print(f"  Valid:   {len(result['valid_citations'])}  {result['valid_citations']}")
    print(f"  Invalid: {len(result['invalid_citations'])}  {result['invalid_citations']}")
    if result["invalid_citations"]:
        print("  WARNING: Hallucinated citations detected!")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # SQLite fix
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    from mmrag.retrieval import Retriever, print_results
    from mmrag.context import ContextBuilder

    retriever = Retriever()
    builder = ContextBuilder()

    query = "Find a chart about comparison across categories. State which category is highest."
    print(f"\nQuery: {query}")

    text_results, image_results = retriever.retrieve(query, top_k=5)
    print_results(text_results, image_results)

    context, valid_ids, image_paths = builder.build(query, text_results, image_results)
    result = generate_answer(query, context, valid_ids, image_paths)
    print_generation(query, result)