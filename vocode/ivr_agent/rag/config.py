from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class RagConfig:
    enabled: bool = False
    provider: str = "none"
    collection: Optional[str] = None
    top_k: int = 4
    max_chars: int = 2000


def format_rag_context(chunks: Iterable[str], config: RagConfig) -> str:
    """Format retrieved chunks into a bounded context block for the LLM."""
    if not config.enabled:
        return ""

    joined = "\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
    if len(joined) > config.max_chars:
        joined = joined[: config.max_chars].rstrip()
    if not joined:
        return ""
    return f"Context:\n{joined}"


def get_rag_context(query: str, config: RagConfig) -> str:
    """Placeholder for retrieval; returns empty until a real backend is wired."""
    if not config.enabled:
        return ""
    # TODO: wire to vector DB or retrieval provider.
    _ = query
    return ""
