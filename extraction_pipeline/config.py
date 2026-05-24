import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BATCH_MODEL: str = os.getenv("OPENAI_BATCH_MODEL", "gpt-4o-mini")

DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/extraction_pipeline")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MAX_RETRIES: int = 3
MAX_TOKENS_PER_CHUNK: int = 6000
BATCH_SIZE: int = 100
BATCH_POLL_INTERVAL_SECS: int = 10
BATCH_POLL_MAX_SECS: int = 60

CONFIDENCE_AUTO_APPROVE: float = 0.90
CONFIDENCE_QA_SAMPLE: float = 0.65

SLA_TARGET_P95_SECS: int = 30

CRITICAL_FIELDS: list[str] = ["invoice_total", "document_type", "author"]
