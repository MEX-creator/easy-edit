"""Exception hierarchy for EditDNA."""

from __future__ import annotations


class EditDNAError(Exception):
    """Base class for all EditDNA errors."""


class ConfigError(EditDNAError):
    """Bad or missing configuration (env vars, .env file)."""


class MissingDependency(EditDNAError):
    """A required optional dependency is not installed."""

    def __init__(self, feature: str, pip_extra: str) -> None:
        self.feature = feature
        self.pip_extra = pip_extra
        super().__init__(
            f"{feature} requires extra dependencies. "
            f"Install with: pip install editdna[{pip_extra}]"
        )


class AnalysisError(EditDNAError):
    """Video/audio analysis failed (no audio track, corrupt file, ...)."""


class TemplateError(EditDNAError):
    """Style template file is invalid or incompatible."""


class EdlError(EditDNAError):
    """Edit Decision List is invalid or could not be planned."""


class LLMError(EditDNAError):
    """LLM call failed (network, quota, malformed JSON)."""


class ResolveError(EditDNAError):
    """DaVinci Resolve scripting bridge failure."""
