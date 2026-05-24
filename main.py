"""
CLI entry point for the Structured Data Extraction Pipeline.

Usage:
  python main.py --demo              Run single-doc extraction demo (requires OPENAI_API_KEY)
  python main.py --retry-demo        Demonstrate retry loop with simulated bad date
  python main.py --few-shot-compare  Compare extraction with/without few-shot examples
  python main.py --batch             Run a 5-doc batch (requires OPENAI_API_KEY)
  python main.py --benchmark         SLA benchmark on 100 mock results (no API key needed)
  python main.py --architecture      Print ASCII architecture diagram
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time

from extraction_pipeline.batch.sla_analyzer import SLAAnalyzer
from extraction_pipeline.routing.confidence_router import ConfidenceRouter
from extraction_pipeline.schemas.batch import BatchJob, BatchResult
from extraction_pipeline.schemas.extraction import (
    ConfidenceScores,
    DocumentType,
    ExtractedDocument,
)
from extraction_pipeline.tests.conftest import make_mock_engine
from extraction_pipeline.validation.orchestrator import ValidationRetryOrchestrator


ARCHITECTURE_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          STRUCTURED DATA EXTRACTION PIPELINE — ARCHITECTURE                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Input Documents (PDFs, text, scans)                                         ║
║       │                                                                      ║
║       ▼                                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │            ValidationRetryOrchestrator                               │    ║
║  │   ┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐  │    ║
║  │   │FewShotLibrary│   │ExtractionEngine│   │Chunk if oversized    │  │    ║
║  │   │ 6 examples   │──►│OpenAI Responses│   │(>6000 tokens)        │  │    ║
║  │   │ (keyword sel)│   │API tool-use    │   └──────────────────────┘  │    ║
║  │   └──────────────┘   │strict=True     │                             │    ║
║  │                       └───────┬───────┘                             │    ║
║  │                               │ raw_dict                            │    ║
║  │                               ▼                                     │    ║
║  │              ┌────────────────────────────────┐                     │    ║
║  │              │    3-Layer Validation Stack     │                     │    ║
║  │              │  1. JSON Schema (Draft 2020-12) │                     │    ║
║  │              │  2. Pydantic v2 model_validate  │                     │    ║
║  │              │  3. Semantic business rules     │                     │    ║
║  │              └────────────┬───────────────────┘                     │    ║
║  │                           │ pass / fail                             │    ║
║  │              ┌────────────┴───────────────────┐                     │    ║
║  │              │ fail: classify_failure()        │                     │    ║
║  │              │  retry_resolvable → retry (≤3)  │                     │    ║
║  │              │  non_resolvable  → raise        │                     │    ║
║  │              └─────────────────────────────────┘                    │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║       │ ExtractedDocument                                                    ║
║       ▼                                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │               ConfidenceRouter                                       │    ║
║  │   overall ≥ 0.90 → auto_approve                                      │    ║
║  │   overall 0.65–0.89 → qa_sample                                      │    ║
║  │   overall < 0.65 → human_review                                      │    ║
║  │   + field-level escalation for critical fields                       │    ║
║  └──────────────────┬───────────────────┬────────────────┬─────────────┘    ║
║                     │                   │                │                   ║
║               auto_approve         qa_sample        human_review            ║
║                     │                   │                │                   ║
║               PostgreSQL          Redis ZADD        Redis ZADD               ║
║               (audit_log)         (queue:qa)        (queue:hr)               ║
║                     │                   │                │                   ║
║                     │             FastAPI /queue/qa  FastAPI /queue/hr       ║
║                     │                   │                │                   ║
║                     └───────────────────┴────────────────┘                  ║
║                                         │                                    ║
║                              PostgreSQL analytics DB                         ║
║                              (extraction_runs, field_scores, sla_metrics)   ║
║                                                                              ║
║  Batch Path (100 docs):                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  BatchProcessor → OpenAI Message Batches API                         │    ║
║  │  Poll with exponential backoff → parse output by custom_id           │    ║
║  │  recover_failed() → mini recovery batch for failed custom_ids        │    ║
║  │  SLAAnalyzer (pandas) → P50/P95/P99, throughput/hr, pass_rate        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


async def demo_mode() -> None:
    print("\n── Single Document Extraction Demo ─────────────────────────────")
    from extraction_pipeline.extraction.engine import ExtractionEngine
    import openai
    from extraction_pipeline import config

    if not config.OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set. Set it in .env or environment.")
        return

    client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    orchestrator = ValidationRetryOrchestrator(engine=ExtractionEngine(client))

    documents = [
        ("doc-demo-invoice", "Invoice #INV-001\nDate: 2024-07-15\nInvoice Total: USD 4,592.50"),
        ("doc-demo-paper", "Quantum Computing Review\nBy: Dr. Jane Smith\nPublished: 2023-03-01\nAbstract: This paper reviews...\nReferences:\n[1] Feynman RP. Simulating physics. 1982."),
        ("doc-demo-ambiguous", "GOVERNMENT FILING — Annual Report 2023\nFiled by: Acme Corp. No invoice, no citations."),
    ]

    for doc_id, text in documents:
        try:
            doc, metrics = await orchestrator.extract_with_retry(doc_id, text)
            router = ConfidenceRouter()
            item = router.route(doc)
            print(f"\n[{doc_id}]")
            print(f"  Type      : {doc.document_type.value}")
            print(f"  Author    : {doc.author}")
            print(f"  Date      : {doc.publication_date}")
            print(f"  Total     : {doc.invoice_total} {doc.currency or ''}")
            print(f"  Confidence: {doc.confidence_scores.overall:.2f}")
            print(f"  Route     : {item.route.value}")
            print(f"  Attempts  : {metrics.attempt_count}")
        except Exception as e:
            print(f"  [FAILED] {e}")


async def retry_demo_mode() -> None:
    print("\n── Retry Loop Demo ──────────────────────────────────────────────")

    bad_date_raw = {
        "document_id": "doc-retry-demo",
        "document_type": "research_paper",
        "document_type_other_detail": None,
        "title": "Climate Dynamics Study",
        "author": "Prof. Lee",
        "publication_date": "March 2024",  # bad format
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.85,
            "title": 0.99,
            "author": 0.99,
            "publication_date": 0.85,
            "invoice_total": None,
            "citations": None,
        },
    }
    good_date_raw = dict(bad_date_raw)
    good_date_raw["publication_date"] = "2024-03-01"

    engine = make_mock_engine([bad_date_raw, good_date_raw])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, metrics = await orchestrator.extract_with_retry("doc-retry-demo", "Research paper text.")

    print(f"  Attempt 1 date   : 'March 2024'  [FAIL — not ISO 8601]")
    print(f"  Attempt 2 date   : '{doc.publication_date}'  [PASS]")
    print(f"  Total attempts   : {metrics.attempt_count}")
    print(f"  Retry successes  : {metrics.retry_successes}")
    print(f"  Final status     : {metrics.final_status}")


async def few_shot_compare_mode() -> None:
    print("\n── Few-Shot Comparison Demo ─────────────────────────────────────")

    no_fs_raw = {
        "document_id": "doc-compare",
        "document_type": "research_paper",
        "document_type_other_detail": None,
        "title": "Market Analysis 2023",
        "author": "Bob Chen",
        "publication_date": "2023-06-01",
        "invoice_total": None,
        "currency": None,
        "citations": ["Smith J. 2022."],
        "confidence_scores": {
            "overall": 0.70,
            "title": 0.80,
            "author": 0.80,
            "publication_date": 0.70,
            "invoice_total": None,
            "citations": 0.70,
        },
    }
    with_fs_raw = dict(no_fs_raw)
    with_fs_raw["confidence_scores"] = {
        "overall": 0.92,
        "title": 0.99,
        "author": 0.99,
        "publication_date": 0.92,
        "invoice_total": None,
        "citations": 0.92,
    }

    engine_no = make_mock_engine([no_fs_raw])
    engine_with = make_mock_engine([with_fs_raw])

    orch_no = ValidationRetryOrchestrator(engine=engine_no)
    orch_with = ValidationRetryOrchestrator(engine=engine_with)

    doc_no, _ = await orch_no.extract_with_retry("doc-compare", "Research paper text.")
    doc_with, _ = await orch_with.extract_with_retry("doc-compare", "Research paper text.")

    improvement = doc_with.confidence_scores.overall - doc_no.confidence_scores.overall
    print(f"  Without few-shot: overall confidence = {doc_no.confidence_scores.overall:.2f}")
    print(f"  With few-shot   : overall confidence = {doc_with.confidence_scores.overall:.2f}")
    print(f"  Improvement     : +{improvement:.2f} ({'✓ ≥0.10' if improvement >= 0.10 else '✗ <0.10'})")


async def benchmark_mode() -> None:
    print("\n── SLA Benchmark (100 mock documents) ──────────────────────────")
    rng = random.Random(42)
    results = [
        BatchResult(
            custom_id=f"doc-{i:03d}",
            document_id=f"doc-{i:03d}",
            success=True,
            latency_ms=max(500, int(rng.gauss(8000, 3000))),
        )
        for i in range(100)
    ]
    job = BatchJob(
        batch_id="benchmark-001",
        status="completed",
        submitted_count=100,
        completed_count=100,
        failed_count=0,
        results=results,
    )
    report = SLAAnalyzer().analyze(job)
    print(report.print_table())

    # Sequential vs Batch comparison
    seq_time_s = sum(r.latency_ms for r in results) / 1000
    batch_time_s = max(r.latency_ms for r in results) / 1000
    print(f"  Comparison: Sequential vs Batch API")
    print(f"  {'Metric':<25} {'Sequential':>15} {'Batch API':>15}")
    print(f"  {'─'*55}")
    print(f"  {'Total time (s)':<25} {seq_time_s:>15.1f} {batch_time_s:>15.1f}")
    print(f"  {'Throughput/hr':<25} {100/(seq_time_s/3600):>15.1f} {report.throughput_per_hr:>15.1f}")
    print(f"  {'Speedup':<25} {'1.0x':>15} {seq_time_s/batch_time_s:>14.1f}x")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured Data Extraction Pipeline CLI")
    parser.add_argument("--demo", action="store_true", help="Run live extraction demo (requires OPENAI_API_KEY)")
    parser.add_argument("--retry-demo", action="store_true", help="Demonstrate retry loop correction")
    parser.add_argument("--few-shot-compare", action="store_true", help="Compare extraction accuracy with/without few-shot")
    parser.add_argument("--batch", action="store_true", help="Run 5-doc batch (requires OPENAI_API_KEY)")
    parser.add_argument("--benchmark", action="store_true", help="SLA benchmark on 100 mock results")
    parser.add_argument("--architecture", action="store_true", help="Print architecture diagram")
    args = parser.parse_args()

    if args.architecture:
        print(ARCHITECTURE_DIAGRAM)
        return

    if args.demo:
        asyncio.run(demo_mode())
    elif args.retry_demo:
        asyncio.run(retry_demo_mode())
    elif args.few_shot_compare:
        asyncio.run(few_shot_compare_mode())
    elif args.benchmark:
        asyncio.run(benchmark_mode())
    else:
        print("Structured Data Extraction Pipeline")
        print("Use --help to see available commands.")
        print("\nRunning all demos (no API key required for most):")
        asyncio.run(retry_demo_mode())
        asyncio.run(few_shot_compare_mode())
        asyncio.run(benchmark_mode())


if __name__ == "__main__":
    main()
