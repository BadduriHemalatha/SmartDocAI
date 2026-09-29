"""Application configuration loaded from environment variables or Streamlit Secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of the smartdoc package).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _secret_or_env(name: str, default: str = "") -> str:
    """Read a setting from Streamlit Secrets first, then environment variables."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value is not None:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    raw = _secret_or_env(name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {raw!r}"
        ) from exc


def _float_env(name: str, default: float) -> float:
    raw = _secret_or_env(name)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be a number, got {raw!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    max_file_size_mb: int
    max_context_chars: int
    log_level: str
    project_root: Path
    log_dir: Path
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key.strip())


def load_settings() -> Settings:
    root = _PROJECT_ROOT

    return Settings(
        gemini_api_key=_secret_or_env("GEMINI_API_KEY"),
        gemini_model=_secret_or_env(
            "GEMINI_MODEL", "gemini-3.5-flash-lite"
        ),
        embedding_model=_secret_or_env(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=_int_env("CHUNK_SIZE", 900),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 150),
        top_k=_int_env("TOP_K", 5),
        min_similarity=_float_env("MIN_SIMILARITY", 0.20),
        max_file_size_mb=_int_env("MAX_FILE_SIZE_MB", 20),
        max_context_chars=_int_env("MAX_CONTEXT_CHARS", 12000),
        log_level=_secret_or_env("LOG_LEVEL", "INFO").upper(),
        project_root=root,
        log_dir=root / "logs",
    )


settings = load_settings()