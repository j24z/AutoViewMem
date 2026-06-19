#!/usr/bin/env python3
"""
Unified configuration for the standalone AutoViewMem package.

Merges the two configs that lived in the research repo
(mem_rl/config.py + mem_rl/multi_layer/config.py) into a single source of
truth, and adds package-local path constants for build/query state and
results so the build and query stages always agree on where adaptive_views.json
lives.

All values are read from the package-local .env (falling back to the process
environment), with defaults matching the working local services:
  LLM       http://localhost:9100/v1   (model Qwen3-8B)
  embedding http://localhost:9000/v1   (model intfloat/e5-base-v2, dim 768)
  Qdrant    localhost:6333
"""
import os
from dotenv import load_dotenv

_PKG = os.path.dirname(os.path.abspath(__file__))

# Load .env from this package dir regardless of CWD, then fall back to any .env
# in the process environment / cwd.
load_dotenv(os.path.join(_PKG, ".env"))
load_dotenv()

# Disable mem0 posthog telemetry BEFORE anything imports mem0 (offline env hangs
# on the posthog network call at import time otherwise).
os.environ.setdefault("MEM0_TELEMETRY", "False")

# --- Unified LLM configuration ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 8192))

ARK_API_KEY = os.getenv("ARK_API_KEY", "any")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "http://localhost:9100/v1")
ARK_MODEL = os.getenv("ARK_MODEL", "Qwen3-8B")
MODEL_NAME = ARK_MODEL  # backward-compat alias

# --- Embedding configuration ---
# e5-base-v2 is NON-matryoshka: embedding_dims stays 768 and must not be sent as
# a `dimensions` override to the endpoint (vLLM returns HTTP 400). mem0's OpenAI
# embedder only forwards it via embedding_dims in the embedder config, which is
# the path the build/query code already uses.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9000/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "empty")
EMBEDDING_NAME = EMBEDDING_MODEL  # for backward compatibility
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 768))

# --- Data (shared LoCoMo benchmark; not copied into the package) ---
_DATA_DIR = os.getenv("LOCOMO_DIR", "/ossfs/workspace/data/locomo")
DATA_PATH = os.path.join(_DATA_DIR, "locomo.json")
DATA_TRAIN_PATH = os.path.join(_DATA_DIR, "locomo_train.json")
DATA_VAL_PATH = os.path.join(_DATA_DIR, "locomo_val.json")
DATA_TEST_PATH = os.path.join(_DATA_DIR, "locomo_test.json")

# --- Default Qdrant collection (build and query defaults MUST agree) ---
COLLECTION_NAME = "autoviewmem_l1_locomo"

# --- Package-local runtime paths (single source of truth) ---
STATE_DIR = os.path.join(_PKG, "state")
RESULTS_DIR = os.path.join(_PKG, "results")
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Discovery/Convergence artefacts: written by the build stage, read by the query
# stage. Keeping both stages pointed at these constants is what keeps them in
# sync (otherwise the query stage silently falls back to load_views_from_qdrant).
VIEWS_PATH = os.path.join(STATE_DIR, "adaptive_views.json")
CANDIDATE_POOL_PATH = os.path.join(STATE_DIR, "candidate_pool.json")
