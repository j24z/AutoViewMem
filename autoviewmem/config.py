import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Unified LLM configuration. The defaults are placeholders for a local
# OpenAI-compatible endpoint; set real values in your environment or .env.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
ARK_API_KEY = os.getenv("ARK_API_KEY", "any")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "http://localhost:8000/v1")
ARK_MODEL = os.getenv("ARK_MODEL", "qwen3-8b")
MODEL_NAME = ARK_MODEL

# Dataset paths are intentionally relative. Download/preprocess datasets under
# data/locomo instead of committing raw benchmark data or local paths.
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "locomo"))
DATA_PATH = str(DATA_DIR / "locomo.json")
DATA_TRAIN_PATH = str(DATA_DIR / "locomo_train.json")
DATA_VAL_PATH = str(DATA_DIR / "locomo_val.json")
DATA_TEST_PATH = str(DATA_DIR / "locomo_test.json")

# Embedding configuration.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "empty")
EMBEDDING_NAME = EMBEDDING_MODEL
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "768"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autoviewmem_demo")

# Retrieval defaults used by legacy/core helpers.
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "25"))
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "25"))
STRUCTURED_TOP_K = int(os.getenv("STRUCTURED_TOP_K", "10"))
ENABLE_PLANNING = os.getenv("ENABLE_PLANNING", "true").lower() == "true"
ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "true").lower() == "true"
MAX_REFLECTION_ROUNDS = int(os.getenv("MAX_REFLECTION_ROUNDS", "2"))
ENABLE_PARALLEL_RETRIEVAL = os.getenv("ENABLE_PARALLEL_RETRIEVAL", "true").lower() == "true"
MAX_RETRIEVAL_WORKERS = int(os.getenv("MAX_RETRIEVAL_WORKERS", "3"))
