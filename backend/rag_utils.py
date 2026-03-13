"""Small RAG utilities for index storage paths and context truncation."""

import os
from typing import List

from .config import settings


def index_dir(chat_id: str) -> str:
    """Return persisted index directory for a chat/document session."""
    return os.path.join(settings.CHROMA_PATH, chat_id)


def build_context(nodes) -> str:
    """Concatenate retrieved node text into a bounded-size context block."""
    pieces: List[str] = []
    total_chars = 0
    for node in nodes:
        text = node.get_content() or ""
        if not text:
            continue
        room = settings.RAG_CONTEXT_MAX_CHARS - total_chars
        if room <= 0:
            break
        if len(text) > room:
            text = text[:room]
        pieces.append(text)
        total_chars += len(text)
    return "\n\n".join(pieces)
