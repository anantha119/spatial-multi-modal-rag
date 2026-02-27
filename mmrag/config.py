from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
AI2D_IMAGES_DIR = IMAGES_DIR / "ai2d"
CHARTQA_IMAGES_DIR = IMAGES_DIR / "chartqa"
CHROMA_DIR = DATA_DIR / "chroma_db"
DOCS_DIR = DATA_DIR / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset identifiers (HuggingFace)
AI2D_DATASET = "lmms-lab/ai2d"
CHARTQA_DATASET = "HuggingFaceM4/ChartQA"

# Embedding models
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim
IMAGE_ENCODER = "clip-ViT-B-32"                           # 512-dim (vision + text)

# ChromaDB collections
TEXT_COLLECTION = "mmrag_text"
IMAGE_COLLECTION = "mmrag_image"

# Retrieval
TOP_K = 5
SCORE_THRESHOLD = 0.4  # Below this, flag as low confidence

# Generation
LLM_PROVIDER = "local"
LLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_TOKENS = 1024