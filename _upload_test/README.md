# AutoViewMem (standalone)

A clean, self-contained copy of the **AutoViewMem** paper method, extracted from
the research repo `mem/mem_rl/`. This package contains **only the code the
default-config pipeline actually executes** — no ablation variants, no L0/L2
layers, no routing experiments. It runs against the local services:

| Service   | Endpoint                       | Model                  |
|-----------|--------------------------------|------------------------|
| LLM       | `http://localhost:9100/v1`     | `Qwen3-8B`             |
| Embedding | `http://localhost:9000/v1`     | `intfloat/e5-base-v2`  |
| Qdrant    | `localhost:6333`               | —                      |

Configuration is read from `autoviewmem/.env` (copy `.env.example` and edit).
mem0 is the editable vendored install at `mem/mem_rl/mem0` (mem0ai 1.0.1).

## The method (default config)

**Build** (`experiment_adaptive.py`, three phases):
1. **Discovery** — stream the conversation per speaker; the LLM proposes
   candidate extraction instructions ("view-prompts") into a per-user pool.
2. **Convergence** — pick `k_views=10` orthogonal views. Default `use_dpp=True`
   → greedy **k-DPP** over the embedding kernel; `--no_dpp` →
   `AgglomerativeClustering`. Frozen to `state/adaptive_views.json`.
3. **Extraction** — re-pass the conversation, extract facts under each of the 10
   views, store to the Qdrant collection. Default `enable_consolidation=False`
   (no online dedup).

**Retrieval** (`search.py`) — **defaults: single-layer L1, fixed quota, no
routing.** `--routing_mode fixed` + `--disable_view_routing` (default True) +
`--l0_limit 0` + `--l1_limit 100` ⇒ L0 (raw layer) is never queried, all top-K
come from L1, the LLM layer-routing prompt never fires.

**Eval** (`evaluation/evals.py`) — LLM-judge / F1 / recall@K.

**Offline compaction (optional, separate script)**
(`scripts/deduplicate_memories.py`) — builds a cosine-similarity graph over a
user's stored memories (`nx.connected_components`, `SIMILARITY_THRESHOLD=0.9`),
recursively tightens the threshold to split big clusters, then LLM-merges each
cluster. Writes `source` → `dest` collection.

## Layout

```
autoviewmem/
  config.py              # unified config + path constants (single source of truth)
  log_config.py
  .env / .env.example
  experiment_adaptive.py # BUILD entry  (python -m autoviewmem.experiment_adaptive)
  search.py              # QUERY entry  (python -m autoviewmem.search)
  adaptive/              # adaptive_ingestion.py + schema_discovery.py + utils.py
  evaluation/            # evals.py + prompts.py + metrics/
  scripts/               # deduplicate_memories.py (optional offline compaction)
  state/                 # generated: adaptive_views.json + candidate_pool.json
  results/               # generated: routing_*.json + evaluation_metrics.json
```

## Running

Run from the `mem/` directory (so `autoviewmem` is importable), with the local
services up and Qdrant started from a stable CWD (`services/start_qdrant.sh`):

```bash
cd /ossfs/workspace/mem

# 1) BUILD — Discovery → k-DPP convergence (~10 views) → multi-view extraction
python -m autoviewmem.experiment_adaptive --mode all --collection_name autoviewmem_l1_locomo

# 2) QUERY/QA — single-layer L1, fixed quota, no routing, pure top-100
python -m autoviewmem.search \
  --input /ossfs/workspace/data/locomo/locomo_test.json \
  --l1_collection autoviewmem_l1_locomo \
  --routing_mode fixed --l0_limit 0 --l1_limit 100 \
  --output autoviewmem/results/routing_locomo.json

# 3) EVAL — LLM-judge / F1 / recall@K
python -m autoviewmem.evaluation.evals \
  --input_file autoviewmem/results/routing_locomo.json \
  --output_file autoviewmem/results/evaluation_metrics.json

# 4) (optional) offline similarity-graph clustering + LLM merge
python -m autoviewmem.scripts.deduplicate_memories \
  --source autoviewmem_l1_locomo --dest autoviewmem_l1_locomo_dedup
```

All defaults already encode the default pipeline, so the flags above are
explicit-but-optional except `--input`/`--output`/`--collection_name`.

### Quick smoke test

```bash
# build on the first 2 conversations only
python -m autoviewmem.experiment_adaptive --mode all --test
# → state/adaptive_views.json + state/candidate_pool.json appear,
#   and the autoviewmem_l1_locomo collection shows up in Qdrant.
curl -s http://localhost:6333/collections
```

## Notes / gotchas

- **build resets the collection (with backup)**: `--force` is **on by default**,
  so each build drops + recreates the target collection before rebuilding. To
  avoid silently destroying a previously built store, the build first copies any
  existing non-empty collection to a timestamped backup
  (`<collection>_bak_<YYYYMMDD_HHMMSS>`). Pass `--no_backup` to skip the copy, or
  `--no_force` to skip the reset entirely (note: `--no_force` makes extraction
  *append* to the existing collection). Different `--collection_name` values
  never touch each other — reset only affects the named collection.
- **views path agreement**: build writes and query reads `state/adaptive_views.json`
  via `config.VIEWS_PATH`. If the build hasn't run, query falls back to
  recovering views from the Qdrant collection.
- **telemetry**: `MEM0_TELEMETRY=False` is set in `.env` and forced in `config.py`
  before mem0 is imported (offline env hangs on posthog otherwise).
- **e5 is non-matryoshka**: keep `EMBEDDING_SIZE=768`; do not request a smaller
  embedding dim.
- **eval deps**: `evaluation/metrics/utils.py` pulls in `bert_score`,
  `sentence_transformers`, `rouge_score`, `nltk` (already present in the base
  conda env).
- The full LoCoMo build + QA over all 1540 questions is slow; run the smoke test
  first.
