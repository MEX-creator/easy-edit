"""Pluggable LLM provider layer built on litellm.

Providers: gemini (default free tier), openai, anthropic, ollama,
openai_compatible. If no provider/API key is configured, `get_llm()` returns
None and callers fall back to deterministic heuristics — this module never
fails hard just because the LLM is not wanted.

Prompts are strictly structured-JSON-in / structured-JSON-out: the LLM only
ever receives compact numeric digests (never raw video) and must reply with a
single JSON object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.config import Settings
from core.errors import ConfigError, LLMError, MissingDependency

# litellm model identifiers. Overridable via EDITDNA_MODEL.
MODEL_DEFAULTS = {
    "gemini": "gemini/gemini-2.5-flash",
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-5-haiku-latest",
    "ollama": "ollama/llama3.1",
}

SUPPORTED_PROVIDERS = tuple(MODEL_DEFAULTS) + ("openai_compatible",)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract the first JSON object from an LLM reply."""
    text = (text or "").strip()
    if not text:
        return None
    # strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    return None


@dataclass
class LLM:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    timeout_sec: int = 90
    max_retries: int = 2

    @classmethod
    def from_settings(cls, settings: Settings) -> Optional["LLM"]:
        provider = (settings.llm_provider or "gemini").lower().strip()
        if provider in ("", "none", "off", "false", "0"):
            return None
        if provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"unknown EDITDNA_LLM_PROVIDER {provider!r}; "
                f"expected one of {list(SUPPORTED_PROVIDERS)}"
            )
        if provider == "ollama":
            model = settings.llm_model or settings.ollama_model or "llama3.1"
            return cls(
                provider=provider,
                model=model,
                api_key=settings.api_key,
                base_url=settings.ollama_base_url,
            )
        model = settings.llm_model or MODEL_DEFAULTS.get(provider, "")
        return cls(
            provider=provider,
            model=model,
            api_key=settings.api_key,
            base_url=settings.openai_base_url,
        )

    @property
    def available(self) -> bool:
        try:
            import litellm  # noqa: F401
            return True
        except Exception:
            return False

    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        schema_description: str = "a JSON object",
    ) -> Dict[str, Any]:
        """Send a prompt, demand a single JSON object, return it as a dict."""
        if not self.available:
            raise MissingDependency("LLM calls", "llm")
        import litellm

        litellm.drop_params = True  # ignore params a provider doesn't support
        messages: list[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout_sec,
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.provider == "ollama":
            kwargs["api_base"] = self.base_url or "http://localhost:11434"
            kwargs["format"] = "json"
        elif self.provider == "openai_compatible" and self.base_url:
            kwargs["api_base"] = self.base_url
        if self.provider in ("openai", "gemini", "openai_compatible"):
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = litellm.completion(**kwargs)
                content = (resp.choices[0].message.content or "") if resp.choices else ""
                data = _extract_json(content)
                if data is not None:
                    return data
                last_error = LLMError("model replied without a valid JSON object")
            except Exception as exc:  # network, quota, provider error
                last_error = exc
            if attempt < self.max_retries:
                messages[-1] = {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\nIMPORTANT: reply with ONLY a single valid JSON "
                        f"object ({schema_description}) and no commentary."
                    ),
                }
        raise LLMError(
            f"LLM JSON request failed after {self.max_retries + 1} attempts: {last_error}"
        )


def get_llm(settings: Settings | None = None) -> Optional[LLM]:
    """Build an LLM from settings, or None when no provider is configured."""
    return LLM.from_settings(settings or Settings.from_env())
