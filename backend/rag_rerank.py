"""Optional reranking stage for retrieved nodes before context assembly."""

import importlib
from typing import Iterable

from .config import settings


def rerank_nodes(query: str, nodes: Iterable):
    """Optionally rerank retrieved nodes with Cohere and return best candidates."""
    node_list = list(nodes or [])
    if not node_list:
        return node_list

    if not settings.ENABLE_COHERE_RERANK or not settings.COHERE_API_KEY:
        return node_list

    try:
        module = importlib.import_module("llama_index.postprocessor.cohere_rerank")
        CohereRerank = getattr(module, "CohereRerank")
    except Exception:
        return node_list

    top_n = max(1, min(settings.COHERE_RERANK_TOP_N, len(node_list)))
    try:
        reranker = CohereRerank(
            api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_RERANK_MODEL,
            top_n=top_n,
        )
        reranked = reranker.postprocess_nodes(node_list, query_str=query)
        return list(reranked or node_list)
    except Exception:
        return node_list
