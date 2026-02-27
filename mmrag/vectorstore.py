"""
Vector store ingestion: loads documents from JSON, embeds them,
and inserts into ChromaDB collections.
"""

# Fix for old system SQLite on cluster environments
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from PIL import Image

from mmrag.config import (
    CHROMA_DIR, DOCS_DIR,
    TEXT_COLLECTION, IMAGE_COLLECTION,
)
from mmrag.schema import Document, load_documents
from mmrag.embeddings import TextEncoder, ImageEncoder


def create_client() -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def ingest_text(client: chromadb.PersistentClient, docs: list[Document], encoder: TextEncoder):
    """Embed and insert text documents into the text collection."""
    collection = client.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Check if already populated
    if collection.count() > 0:
        print(f"[VectorStore] Text collection already has {collection.count()} docs. Skipping.")
        return collection

    print(f"[VectorStore] Embedding {len(docs)} text documents...")
    contents = [d.content for d in docs]

    # Batch encode
    embeddings = encoder.encode(contents, batch_size=256)

    # Prepare for ChromaDB insertion
    ids = [d.doc_id for d in docs]
    metadatas = [
        {
            "group_id": d.group_id,
            "source_dataset": d.source_dataset,
            "doc_type": d.doc_type,
            "content": d.content,        # store full text for retrieval
            "image_path": d.image_path,
        }
        for d in docs
    ]

    # ChromaDB has a max batch size, insert in chunks
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )
        print(f"[VectorStore]   Inserted text {end}/{len(ids)}")

    print(f"[VectorStore] Text collection: {collection.count()} documents")
    return collection


def ingest_images(client: chromadb.PersistentClient, docs: list[Document], encoder: ImageEncoder):
    """Embed and insert image documents into the image collection."""
    collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"[VectorStore] Image collection already has {collection.count()} docs. Skipping.")
        return collection

    print(f"[VectorStore] Embedding {len(docs)} image documents...")

    # Load and encode images in batches
    batch_size = 128
    all_embeddings = []

    for i in range(0, len(docs), batch_size):
        end = min(i + batch_size, len(docs))
        batch_docs = docs[i:end]

        images = []
        for d in batch_docs:
            img = Image.open(d.image_path).convert("RGB")
            images.append(img)

        batch_emb = encoder.encode_images(images, batch_size=batch_size)
        all_embeddings.append(batch_emb)

        # Close images to free memory
        for img in images:
            img.close()

        print(f"[VectorStore]   Embedded images {end}/{len(docs)}")

    import numpy as np
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Prepare metadata
    ids = [d.doc_id for d in docs]
    metadatas = [
        {
            "group_id": d.group_id,
            "source_dataset": d.source_dataset,
            "doc_type": d.doc_type,
            "content": d.content,
            "image_path": d.image_path,
        }
        for d in docs
    ]

    # Insert in chunks
    insert_batch = 5000
    for i in range(0, len(ids), insert_batch):
        end = min(i + insert_batch, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=all_embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )
        print(f"[VectorStore]   Inserted images {end}/{len(ids)}")

    print(f"[VectorStore] Image collection: {collection.count()} documents")
    return collection


def main():
    # Load serialized documents
    print("[VectorStore] Loading documents from disk...")
    text_docs = load_documents(str(DOCS_DIR / "text_documents.json"))
    image_docs = load_documents(str(DOCS_DIR / "image_documents.json"))
    print(f"[VectorStore] Loaded {len(text_docs)} text, {len(image_docs)} image docs")

    # Initialize encoders
    te = TextEncoder()
    ie = ImageEncoder()

    # Initialize ChromaDB
    client = create_client()

    # Ingest
    ingest_text(client, text_docs, te)
    ingest_images(client, image_docs, ie)

    print("\n[Done] Vector store ready.")


if __name__ == "__main__":
    main()