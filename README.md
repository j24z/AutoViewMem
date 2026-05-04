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

## Commands

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
