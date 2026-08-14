"""Configuration handling: .env loading + typed Settings.

The loader is dependency-free (no python-dotenv required): it parses KEY=VALUE
lines from a `.env` file in the current directory, only setting variables that
are not already present in the environment (real env vars win).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PREFIX = "EDITDNA_"


def load_dotenv_file(path: str | Path | None = None) -> None:
    """Minimal .env loader. Never overrides variables already in the env."""
    p = Path(path) if path else Path(".env")
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _get(key: str, default: str = "") -> str:
    return os.environ.get(PREFIX + key, default)


@dataclass
class Settings:
    llm_provider: str = "gemini"
    api_key: str = ""
    llm_model: str = ""
    openai_base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_min_confidence: float = 0.5
    templates_dir: str = "templates"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file()
        languages = [
            lang
            for lang in _get("OCR_LANGUAGES", "en").replace(" ", "").split(",")
            if lang
        ]
        return cls(
            llm_provider=(_get("LLM_PROVIDER", "gemini") or "gemini").lower().strip(),
            api_key=_get("API_KEY"),
            llm_model=_get("MODEL"),
            openai_base_url=_get("OPENAI_BASE_URL"),
            ollama_base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_get("OLLAMA_MODEL", "llama3.1"),
            ocr_languages=languages or ["en"],
            ocr_min_confidence=float(_get("OCR_MIN_CONFIDENCE", "0.5")),
            templates_dir=_get("TEMPLATES_DIR", "templates"),
        )


def templates_dir(settings: Settings | None = None) -> Path:
    """Resolve the directory where style templates live.

    Priority: EDITDNA_TEMPLATES_DIR → ./templates (if it exists) →
    ~/.editdna/templates (created on demand).
    """
    s = settings or Settings.from_env()
    d = Path(s.templates_dir).expanduser()
    if d.is_absolute():
        d.mkdir(parents=True, exist_ok=True)
        return d
    if d.is_dir():
        return d.resolve()
    home = Path.home() / ".editdna" / "templates"
    home.mkdir(parents=True, exist_ok=True)
    return home
