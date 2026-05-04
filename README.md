# AutoViewMem

Official research code for **AutoViewMem: Self-Configuring Orthogonal Views for Conversational Long-Term Memory**.

This repository contains the minimal source code needed to build adaptive memory views, run retrieval, and evaluate outputs. It intentionally excludes private configuration, raw benchmark data, generated memories, logs, figures, paper drafts, and local experiment artifacts.

## Structure

- `autoviewmem/config.py`: environment-driven runtime configuration.
- `autoviewmem/memory/adaptive/`: adaptive view discovery and memory ingestion.
- `autoviewmem/memory/build.py`: memory construction pipeline.
- `autoviewmem/memory/search.py`: AutoViewMem retrieval and answer generation.
- `autoviewmem/evaluation/`: answer and retrieval metrics.
- `autoviewmem/datasets/locomo/`: LoCoMo preprocessing utilities only.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your own OpenAI-compatible LLM endpoint, embedding endpoint, and Qdrant collection names. Do not commit `.env`.

Start Qdrant locally:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Place benchmark files under `data/locomo/` after obtaining them from the original dataset source.

## Reproduction

The expected LoCoMo layout after preprocessing is:

```text
data/locomo/
  locomo.json
  locomo_train.json
  locomo_val.json
  locomo_test.json
```

Prepare LoCoMo files:

```bash
python -m autoviewmem.datasets.locomo.filter_dataset --input data/locomo/locomo10.json --output data/locomo/locomo.json
python -m autoviewmem.datasets.locomo.split_dataset --input data/locomo/locomo.json --output_dir data/locomo
python -m autoviewmem.datasets.locomo.prepare_qa --input data/locomo/locomo.json --output data/locomo/locomo_qa.json
```

Build AutoViewMem memories:

```bash
python -m autoviewmem.memory.build \
  --input data/locomo/locomo_train.json \
  --collection_name autoviewmem_demo \
  --mode ingest
```

Run retrieval and answer generation:

```bash
python -m autoviewmem.memory.search \
  --input data/locomo/locomo_test.json \
  --l1_collection autoviewmem_demo \
  --output outputs/autoviewmem_results.json
```

Evaluate generated answers:

```bash
python -m autoviewmem.evaluation.evaluate \
  --input_file outputs/YYYYMMDD/autoviewmem_results.json \
  --output_file results/evaluation_metrics.json
```

Use `--enable_llm_judge` only when your `.env` points to a working judge model.

## Command Help

```bash
python -m autoviewmem.memory.build --help
python -m autoviewmem.memory.search --help
python -m autoviewmem.evaluation.evaluate --help
python -m autoviewmem.evaluation.margin_analysis --help
```

## Release Hygiene

This public tree excludes:

- API keys, `.env`, private endpoints, and local absolute paths
- raw datasets and generated result files
- caches, logs, IDE files, paper PDFs, Docx files, and vendored repositories

## License

MIT License.
