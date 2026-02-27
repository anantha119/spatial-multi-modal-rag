"""
Embedding wrappers for text (MiniLM) and image (CLIP) encoders.
Provides a unified interface for encoding queries, text documents, and images.
"""

from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

from mmrag.config import TEXT_ENCODER, IMAGE_ENCODER


class TextEncoder:
    """MiniLM-based encoder for text documents and text queries."""

    def __init__(self, model_name: str = TEXT_ENCODER):
        print(f"[TextEncoder] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[TextEncoder] Ready. Dimension: {self.dim}")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of strings. Returns (N, dim) float32 array."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns (dim,) float32 array."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )


class ImageEncoder:
    """CLIP-based encoder for images (vision) and text-to-image queries (text)."""

    def __init__(self, model_name: str = IMAGE_ENCODER):
        print(f"[ImageEncoder] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = 512  # CLIP ViT-B-32 output dimension
        print(f"[ImageEncoder] Ready. Dimension: {self.dim}")

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a list of PIL Images. Returns (N, dim) float32 array."""
        return self.model.encode(
            images,
            batch_size=batch_size,
            show_progress_bar=len(images) > 100,
            normalize_embeddings=True,
        )

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL Image. Returns (dim,) float32 array."""
        return self.model.encode(
            image,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a text query for searching the image collection.
        Uses CLIP's text encoder to project into the joint vision-language space.
        Returns (dim,) float32 array.
        """
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test text encoder
    te = TextEncoder()
    q = te.encode_query("What does this diagram show?")
    print(f"Text query shape: {q.shape}")

    batch = te.encode(["Question: What is A? Answer: Cell", "Question: What is B? Answer: Nucleus"])
    print(f"Text batch shape: {batch.shape}")

    # Test image encoder
    ie = ImageEncoder()
    q2 = ie.encode_query("a bar chart comparing categories")
    print(f"Image query shape: {q2.shape}")

    # Create a dummy image to test vision encoding
    dummy = Image.new("RGB", (224, 224), color="red")
    v = ie.encode_image(dummy)
    print(f"Image embed shape: {v.shape}")

    print("\n[OK] Both encoders working.")