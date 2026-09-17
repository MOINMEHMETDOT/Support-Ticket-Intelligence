"""Configuration, resolved once from the environment at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    ollama_host: str
    ollama_model: str
    data_csv: Path
    as_of_date: datetime | None
    sql_timeout_seconds: int
    sql_max_rows: int

    @property
    def model_name(self) -> str:
        """The model id for the currently selected provider."""
        return {
            "gemini": self.gemini_model,
            "groq": self.groq_model,
            "ollama": self.ollama_model,
        }.get(self.provider, "unknown")


def _parse_as_of(raw: str) -> datetime | None:
    """Parse AS_OF_DATE. Empty means 'derive from the dataset' (see data.loader)."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"AS_OF_DATE={raw!r} is not a recognised datetime. "
        "Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM'."
    )


def load_settings() -> Settings:
    csv_path = Path(_str("DATA_CSV", "data/support_tickets.csv"))
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    return Settings(
        provider=_str("LLM_PROVIDER", "gemini").lower(),
        gemini_api_key=_str("GEMINI_API_KEY"),
        gemini_model=_str("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        groq_api_key=_str("GROQ_API_KEY"),
        groq_model=_str("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ollama_host=_str("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_str("OLLAMA_MODEL", "qwen2.5:7b"),
        data_csv=csv_path,
        as_of_date=_parse_as_of(_str("AS_OF_DATE")),
        sql_timeout_seconds=_int("SQL_TIMEOUT_SECONDS", 5),
        sql_max_rows=_int("SQL_MAX_ROWS", 200),
    )


settings = load_settings()
