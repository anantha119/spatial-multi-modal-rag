"""
Data ingestion pipeline for AI2D and ChartQA datasets.
Downloads, deduplicates images, normalizes to unified Document schema,
saves images to disk, and serializes document lists to JSON.
"""

import hashlib
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

from mmrag.config import (
    AI2D_DATASET, CHARTQA_DATASET,
    AI2D_IMAGES_DIR, CHARTQA_IMAGES_DIR, DOCS_DIR,
)
from mmrag.schema import Document, save_documents


# ---------------------------------------------------------------------------
# Image hashing
# ---------------------------------------------------------------------------

def hash_image(img) -> str:
    """Produce an MD5 hex digest from raw PIL image bytes."""
    return hashlib.md5(img.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# AI2D ingestion
# ---------------------------------------------------------------------------

def ingest_ai2d() -> tuple[list[Document], list[Document]]:
    """
    Download AI2D (lmms-lab/ai2d), deduplicate images, and return
    (text_docs, image_docs) as normalized Document lists.

    Schema per row:
        question : str
        options  : list[str]  (always length 4)
        answer   : str        ("0","1","2","3" â€” index into options)
        image    : PIL.Image
    """
    print("[AI2D] Loading dataset...")
    ds = load_dataset(AI2D_DATASET, split="test")
    print(f"[AI2D] Loaded {len(ds)} rows")

    AI2D_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1: deduplicate images and assign group_ids
    hash_to_group: dict[str, str] = {}   # image_hash -> group_id
    group_counter = 0

    # Also track which rows belong to which group
    row_groups: list[str] = []           # row index -> group_id
    row_hashes: list[str] = []

    print("[AI2D] Hashing images for deduplication...")
    for i, row in enumerate(ds):
        h = hash_image(row["image"])
        row_hashes.append(h)

        if h not in hash_to_group:
            gid = f"ai2d_{group_counter:04d}"
            hash_to_group[h] = gid
            group_counter += 1

        row_groups.append(hash_to_group[h])

    unique_images = len(hash_to_group)
    print(f"[AI2D] {unique_images} unique images from {len(ds)} rows")

    # Pass 2: save unique images and build documents
    text_docs: list[Document] = []
    image_docs: list[Document] = []

    saved_groups: set[str] = set()        # track which images we already saved
    group_text_counter: dict[str, int] = {}  # group_id -> next text index

    for i, row in enumerate(ds):
        gid = row_groups[i]

        # Save image once per group
        if gid not in saved_groups:
            img_path = AI2D_IMAGES_DIR / f"{gid}.png"
            row["image"].save(str(img_path))
            saved_groups.add(gid)

            image_docs.append(Document(
                doc_id=f"{gid}_img",
                group_id=gid,
                doc_type="image",
                source_dataset="ai2d",
                content="",
                image_path=str(img_path),
                metadata={"original_indices": []},
            ))

        # Find the image doc and track original indices
        img_doc = next(d for d in image_docs if d.group_id == gid)
        img_doc.metadata["original_indices"].append(i)

        # Build text record
        answer_idx = int(row["answer"])
        answer_text = row["options"][answer_idx]
        content = f"Question: {row['question']} Answer: {answer_text}"

        txt_idx = group_text_counter.get(gid, 0)
        group_text_counter[gid] = txt_idx + 1

        text_docs.append(Document(
            doc_id=f"{gid}_txt_{txt_idx}",
            group_id=gid,
            doc_type="text",
            source_dataset="ai2d",
            content=content,
            metadata={
                "question": row["question"],
                "answer": answer_text,
                "answer_index": answer_idx,
                "options": row["options"],
                "original_index": i,
            },
        ))

    print(f"[AI2D] Produced {len(text_docs)} text docs, {len(image_docs)} image docs")
    return text_docs, image_docs


# ---------------------------------------------------------------------------
# ChartQA ingestion
# ---------------------------------------------------------------------------

def ingest_chartqa() -> tuple[list[Document], list[Document]]:
    """
    Download ChartQA (HuggingFaceM4/ChartQA), deduplicate images, and return
    (text_docs, image_docs) as normalized Document lists.

    Schema per row:
        query             : str
        label             : list[str]  (always length 1)
        human_or_machine  : ClassLabel (0=human, 1=machine)
        image             : PIL.Image
    """
    print("[ChartQA] Loading dataset...")
    ds_dict = load_dataset(CHARTQA_DATASET)

    # Concatenate all splits
    splits = []
    for split_name in ds_dict:
        splits.append(ds_dict[split_name])
        print(f"[ChartQA]   {split_name}: {len(ds_dict[split_name])} rows")
    ds = concatenate_datasets(splits)
    print(f"[ChartQA] Total: {len(ds)} rows")

    CHARTQA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve the class label feature for human_or_machine
    hm_feature = ds.features.get("human_or_machine")

    # Pass 1: deduplicate images
    hash_to_group: dict[str, str] = {}
    group_counter = 0
    row_groups: list[str] = []

    print("[ChartQA] Hashing images for deduplication...")
    for i, row in enumerate(ds):
        h = hash_image(row["image"])

        if h not in hash_to_group:
            gid = f"chartqa_{group_counter:04d}"
            hash_to_group[h] = gid
            group_counter += 1

        row_groups.append(hash_to_group[h])

        if (i + 1) % 5000 == 0:
            print(f"[ChartQA]   Processed {i + 1}/{len(ds)} rows...")

    unique_images = len(hash_to_group)
    print(f"[ChartQA] {unique_images} unique images from {len(ds)} rows")

    # Pass 2: save images and build documents
    text_docs: list[Document] = []
    image_docs: list[Document] = []

    saved_groups: set[str] = set()
    group_text_counter: dict[str, int] = {}

    for i, row in enumerate(ds):
        gid = row_groups[i]

        if gid not in saved_groups:
            img_path = CHARTQA_IMAGES_DIR / f"{gid}.png"
            row["image"].save(str(img_path))
            saved_groups.add(gid)

            image_docs.append(Document(
                doc_id=f"{gid}_img",
                group_id=gid,
                doc_type="image",
                source_dataset="chartqa",
                content="",
                image_path=str(img_path),
                metadata={"original_indices": []},
            ))

        img_doc = next(d for d in image_docs if d.group_id == gid)
        img_doc.metadata["original_indices"].append(i)

        # Build text record
        answer_text = row["label"][0]
        content = f"Question: {row['query']} Answer: {answer_text}"

        # Resolve human_or_machine label
        hm_raw = row["human_or_machine"]
        if hm_feature and hasattr(hm_feature, "int2str"):
            hm_label = hm_feature.int2str(hm_raw)
        else:
            hm_label = str(hm_raw)

        txt_idx = group_text_counter.get(gid, 0)
        group_text_counter[gid] = txt_idx + 1

        text_docs.append(Document(
            doc_id=f"{gid}_txt_{txt_idx}",
            group_id=gid,
            doc_type="text",
            source_dataset="chartqa",
            content=content,
            metadata={
                "question": row["query"],
                "answer": answer_text,
                "human_or_machine": hm_label,
                "original_index": i,
            },
        ))

        if (i + 1) % 5000 == 0:
            print(f"[ChartQA]   Built docs for {i + 1}/{len(ds)} rows...")

    print(f"[ChartQA] Produced {len(text_docs)} text docs, {len(image_docs)} image docs")
    return text_docs, image_docs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(text_docs: list[Document], image_docs: list[Document]) -> None:
    """Run basic sanity checks on the produced documents."""
    print("\n[Validate] Running checks...")

    # Every text doc has content
    empty_text = [d for d in text_docs if not d.content.strip()]
    assert len(empty_text) == 0, f"{len(empty_text)} text docs have empty content"

    # Every image doc has a valid path
    missing_imgs = [d for d in image_docs if not Path(d.image_path).exists()]
    assert len(missing_imgs) == 0, f"{len(missing_imgs)} image docs have missing files"

    # Every doc has a group_id
    no_group = [d for d in text_docs + image_docs if not d.group_id]
    assert len(no_group) == 0, f"{len(no_group)} docs have no group_id"

    # Every text doc has a corresponding image doc with the same group_id
    image_groups = {d.group_id for d in image_docs}
    orphan_text = [d for d in text_docs if d.group_id not in image_groups]
    assert len(orphan_text) == 0, f"{len(orphan_text)} text docs have no matching image"

    # No duplicate doc_ids
    all_ids = [d.doc_id for d in text_docs + image_docs]
    dupes = len(all_ids) - len(set(all_ids))
    assert dupes == 0, f"{dupes} duplicate doc_ids found"

    print("[Validate] All checks passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Ingest AI2D
    ai2d_text, ai2d_images = ingest_ai2d()
    validate(ai2d_text, ai2d_images)

    # Ingest ChartQA
    cqa_text, cqa_images = ingest_chartqa()
    validate(cqa_text, cqa_images)

    # Merge and save
    all_text = ai2d_text + cqa_text
    all_images = ai2d_images + cqa_images

    save_documents(all_text, str(DOCS_DIR / "text_documents.json"))
    save_documents(all_images, str(DOCS_DIR / "image_documents.json"))

    print(f"\n[Done] Total text docs:  {len(all_text)}")
    print(f"[Done] Total image docs: {len(all_images)}")
    print(f"[Done] Saved to {DOCS_DIR}")


if __name__ == "__main__":
    main()