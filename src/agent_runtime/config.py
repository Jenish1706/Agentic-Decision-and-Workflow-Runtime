"""Central configuration. One .env file, one Settings object, sane defaults
for everything so the project runs out of the box with no configuration at
all - .env only needs to override what you actually want to change."""

import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# sentence-transformers/huggingface_hub log every HTTP HEAD request they make
# to check for model updates at INFO level, which drowns out this project's
# own trace output. Quieting them here (imported before anything else loads
# a model) keeps that noise out of demo scripts without disabling the checks.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _noisy_logger in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM backend (Ollama, local)
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k_retrieve: int = 8
    top_k_rerank: int = 3

    # Agent
    max_agent_steps: int = 8
    max_decision_retries: int = 2

    # Paths
    documents_dir: Path = DATA_DIR / "documents"
    records_path: Path = DATA_DIR / "records.json"
    benchmark_path: Path = DATA_DIR / "benchmark.json"
    long_term_memory_path: Path = DATA_DIR / "long_term_memory.json"


settings = Settings()
