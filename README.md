# Structured Data Extraction Pipeline

An AI-powered document extraction pipeline that uses OpenAI's tool-use API to extract structured data from unstructured documents (invoices, research papers, and more). Features a 3-layer validation stack, confidence-based routing, retry orchestration, and batch processing with SLA analytics.

## Architecture

```
Input Documents (PDFs, text, scans)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│            ValidationRetryOrchestrator                               │
│   ┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐  │
│   │FewShotLibrary│   │ExtractionEngine│   │Chunk if oversized    │  │
│   │ 6 examples   │──►│OpenAI tool-use │   │(>6000 tokens)        │  │
│   │ (keyword sel)│   │strict=True     │   └──────────────────────┘  │
│   └──────────────┘   └───────┬───────┘                             │
│                               │ raw_dict                            │
│                               ▼                                     │
│              ┌────────────────────────────────┐                     │
│              │    3-Layer Validation Stack     │                     │
│              │  1. JSON Schema (Draft 2020-12) │                     │
│              │  2. Pydantic v2 model_validate  │                     │
│              │  3. Semantic business rules     │                     │
│              └────────────┬───────────────────┘                     │
│                           │ pass / fail                             │
│              ┌────────────┴───────────────────┐                     │
│              │ fail: classify_failure()        │                     │
│              │  retry_resolvable → retry (≤3)  │                     │
│              │  non_resolvable  → raise        │                     │
│              └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
       │ ExtractedDocument
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│               ConfidenceRouter                                       │
│   overall ≥ 0.90 → auto_approve                                      │
│   overall 0.65–0.89 → qa_sample                                      │
│   overall < 0.65 → human_review                                      │
│   + field-level escalation for critical fields                       │
└──────────────────┬───────────────────┬────────────────┬─────────────┘
                   │                   │                │
             auto_approve         qa_sample        human_review
                   │                   │                │
             PostgreSQL          Redis ZADD        Redis ZADD
             (audit_log)         (queue:qa)        (queue:hr)
```

**Batch path** (100 docs): `BatchProcessor` → OpenAI Message Batches API → poll with exponential backoff → recover failed items → `SLAAnalyzer` (pandas) reports P50/P95/P99 latency, throughput/hr, and pass rate.

## Features

- **Tool-use extraction** — OpenAI function calling with `strict=True` for guaranteed schema adherence
- **Few-shot selection** — keyword-based retrieval of up to 6 examples to improve accuracy
- **3-layer validation** — JSON Schema + Pydantic v2 + semantic business rules
- **Smart retry** — classifies failures as resolvable (retry) or non-resolvable (raise), up to 3 attempts
- **Chunking** — documents exceeding 6,000 tokens are split and merged automatically
- **Confidence routing** — auto-approve, QA sample, or human review based on per-field scores
- **Batch processing** — OpenAI Message Batches API with recovery for failed items
- **SLA analytics** — P50/P95/P99 latency, throughput/hr, pass rate via pandas
- **Review queues** — FastAPI endpoints backed by Redis sorted sets
- **Audit logging** — PostgreSQL audit trail for all extraction runs

## Project Structure

```
extraction_pipeline/
├── extraction/         # ExtractionEngine, prompts, few-shot library, tool schema
├── validation/         # Pydantic validator, JSON schema validator, retry orchestrator
├── routing/            # ConfidenceRouter (auto_approve / qa_sample / human_review)
├── batch/              # BatchProcessor, BatchPoller, SLAAnalyzer
├── review/             # ReviewQueue, FastAPI review endpoints
├── storage/            # PostgreSQL (db.py, audit_log.py), analytics
├── schemas/            # Pydantic models (ExtractedDocument, BatchJob, etc.)
├── tests/              # pytest test suite
└── config.py           # All tunable constants (thresholds, URLs, limits)
main.py                 # CLI entry point
```

## Requirements

- Python 3.11+
- PostgreSQL (for audit log and analytics)
- Redis (for review queues)
- OpenAI API key (for live extraction; most demos run without one)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY, DATABASE_URL, REDIS_URL
```

## Usage

```bash
# Show architecture diagram
python main.py --architecture

# Run single-document extraction (requires OPENAI_API_KEY)
python main.py --demo

# Demonstrate retry loop with a simulated bad date field
python main.py --retry-demo

# Compare extraction accuracy with and without few-shot examples
python main.py --few-shot-compare

# Run a 5-document batch (requires OPENAI_API_KEY)
python main.py --batch

# SLA benchmark on 100 mock results (no API key needed)
python main.py --benchmark

# Run all non-API demos
python main.py
```

## Configuration

Key constants in `extraction_pipeline/config.py`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o` | Model for single-doc extraction |
| `OPENAI_BATCH_MODEL` | `gpt-4o-mini` | Model for batch processing |
| `MAX_RETRIES` | `3` | Max validation retry attempts |
| `MAX_TOKENS_PER_CHUNK` | `6000` | Token limit before chunking |
| `CONFIDENCE_AUTO_APPROVE` | `0.90` | Threshold for auto-approval |
| `CONFIDENCE_QA_SAMPLE` | `0.65` | Threshold for QA vs human review |
| `SLA_TARGET_P95_SECS` | `30` | P95 latency SLA target |
| `CRITICAL_FIELDS` | `invoice_total, document_type, author` | Fields that trigger escalation |

## Running Tests

```bash
pytest
```

Tests cover: confidence routing, few-shot accuracy, retry/correction logic, batch recovery, chunking retry, null field handling, enum fallbacks, non-resolvable failures, and SLA benchmarks.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For live extraction | OpenAI API key |
| `OPENAI_MODEL` | No | Override default model (`gpt-4o`) |
| `OPENAI_BATCH_MODEL` | No | Override batch model (`gpt-4o-mini`) |
| `DATABASE_URL` | For persistence | PostgreSQL connection string |
| `REDIS_URL` | For review queues | Redis connection string |
# Structured-Data-Extraction-Pipeline
