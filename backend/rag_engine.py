"""Core RAG ingestion and response streaming engine."""

import os
import re
import shutil
from dataclasses import dataclass
from typing import Callable, Iterable, List

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from .config import settings
from .model_catalog import build_langchain_llm
from .rag_embedding import ensure_embedding_model
from .rag_loaders import build_splitter, load_documents
from .rag_prompting import build_rag_messages
from .rag_rerank import rerank_nodes
from .rag_utils import build_context, index_dir


def _to_text_chunk(value) -> str:
    """Normalize different streaming payload shapes into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _to_text_chunk(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content"):
            payload = value.get(key)
            if payload is not None:
                return _to_text_chunk(payload)
        return ""
    content = getattr(value, "content", None)
    if content is not None:
        return _to_text_chunk(content)
    return str(value)


def _is_history_meta_query(query: str) -> bool:
    """Detect whether user query asks about prior conversation turns."""
    q = (query or "").strip().lower()
    if not q:
        return False
    markers = (
        "what was my last question",
        "what did i ask",
        "what did i ask last",
        "my previous question",
        "repeat my last question",
        "what was your last answer",
        "your previous answer",
        "what did you answer",
        "what were we discussing",
        "summarize our conversation",
    )
    return any(m in q for m in markers)


def _answer_history_meta_query(query: str, history: List[dict] | None) -> str:
    """Answer conversation-meta prompts directly from in-memory chat history."""
    turns = history or []
    if not turns:
        return "I do not know based on the uploaded document."

    q = (query or "").strip().lower()
    last_turn = turns[-1]
    last_user = (last_turn.get("user") or "").strip()
    last_assistant = (last_turn.get("assistant") or "").strip()

    if "last question" in q or "did i ask" in q or "previous question" in q:
        return (
            f"Your last question was: \"{last_user}\""
            if last_user
            else "I do not know based on the uploaded document."
        )
    if "last answer" in q or "did you answer" in q or "previous answer" in q:
        return (
            f"My last answer was: \"{last_assistant}\""
            if last_assistant
            else "I do not know based on the uploaded document."
        )
    if "summarize our conversation" in q or "what were we discussing" in q:
        lines: list[str] = []
        for turn in turns[-3:]:
            user_text = (turn.get("user") or "").strip()
            assistant_text = (turn.get("assistant") or "").strip()
            if user_text:
                lines.append(f"User asked: {user_text}")
            if assistant_text:
                lines.append(f"Assistant replied: {assistant_text}")
        return "\n".join(lines) if lines else "I do not know based on the uploaded document."
    return "I do not know based on the uploaded document."


def _is_summary_query(query: str) -> bool:
    """Detect whether query asks for a summary or overview."""
    q = (query or "").strip().lower()
    if not q:
        return False
    # If the prompt also asks for calculations/analytics, do not force summary mode.
    analytic_markers = (
        "total",
        "sum",
        "count",
        "average",
        "mean",
        "max",
        "min",
        "calculate",
        "transaction value",
        "how many",
        "compare",
        "difference",
        "in 20",
        " in 19",
    )
    if any(m in q for m in analytic_markers) or q.count("?") > 1:
        return False
    markers = (
        "summarize",
        "summary",
        "summarise",
        "overview",
        "what is in this image",
        "what's in this image",
        "contents of the image",
        "contents of this image",
        "summarize contents",
        "summarize this image",
    )
    return any(m in q for m in markers)


def _extractive_context_summary(context: str, max_lines: int = 8) -> str:
    """Build a concise bullet summary directly from retrieved context lines."""
    lines: list[str] = []
    seen: set[str] = set()

    for raw in (context or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" -\t")
        if not line:
            continue
        if line.lower().startswith("page ") or line.lower().startswith("[page "):
            continue
        alpha_count = sum(ch.isalpha() for ch in line)
        if alpha_count < 4:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= max_lines:
            break

    if not lines:
        return ""
    bullets = "\n".join(f"- {line}" for line in lines)
    return "Summary from extracted document/image text:\n" + bullets


@dataclass(frozen=True)
class RAGDependencies:
    ensure_embedding_model_fn: Callable[[], object] = ensure_embedding_model
    load_documents_fn: Callable[[str], list] = load_documents
    build_splitter_fn: Callable[[str], object] = build_splitter
    index_dir_fn: Callable[[str], str] = index_dir
    build_context_fn: Callable[[Iterable], str] = build_context
    build_messages_fn: Callable[[str, str, List[dict] | None], List[tuple]] = build_rag_messages
    rerank_nodes_fn: Callable[[str, Iterable], list] = rerank_nodes
    build_llm_fn: Callable[[str, float], object] = build_langchain_llm
    rag_top_k: int = settings.RAG_TOP_K


class RAGEngine:
    def __init__(self, deps: RAGDependencies | None = None):
        """Initialize the RAG engine with concrete dependency bindings."""
        self.deps = deps or RAGDependencies()

    def ingest_document(self, file_path: str, chat_id: str):
        """Parse, chunk, embed, and persist an uploaded document index."""
        embed_model = self.deps.ensure_embedding_model_fn()
        documents = self.deps.load_documents_fn(file_path)
        splitter = self.deps.build_splitter_fn(file_path)
        nodes = splitter.get_nodes_from_documents(documents)
        if not nodes:
            raise RuntimeError("No text chunks were generated from the uploaded file.")

        persist_dir = self.deps.index_dir_fn(chat_id)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
        index.storage_context.persist(persist_dir=persist_dir)

    def stream_rag_response(
        self,
        chat_id: str,
        query: str,
        model: str,
        temperature: float,
        history: List[dict] | None = None,
    ):
        """Retrieve context and stream model output for a chat question."""
        try:
            if _is_history_meta_query(query):
                yield _answer_history_meta_query(query, history)
                return

            nodes = self._retrieve_nodes(chat_id, query)
            context = self.deps.build_context_fn(nodes)

            if not context.strip():
                yield "I could not find relevant content in the uploaded document."
                return

            messages = self.deps.build_messages_fn(context, query, history)
            llm = self.deps.build_llm_fn(model, temperature)

            if _is_summary_query(query):
                summary_response = llm.invoke(messages)
                summary_text = _to_text_chunk(summary_response).strip()
                if (
                    not summary_text
                    or "i do not know based on the uploaded document" in summary_text.lower()
                ):
                    extractive = _extractive_context_summary(context)
                    if extractive:
                        yield extractive
                    else:
                        yield "I do not know based on the uploaded document."
                else:
                    yield summary_text
                return

            emitted = False
            for chunk in llm.stream(messages):
                text = _to_text_chunk(chunk)
                if text:
                    emitted = True
                    yield text
            if not emitted:
                # Some providers return non-text/empty streaming chunks on follow-up turns.
                # Fallback to non-stream call to avoid blank responses.
                response = llm.invoke(messages)
                text = _to_text_chunk(response)
                if text:
                    yield text
                else:
                    yield "I could not generate a response for this prompt."
        except Exception as e:
            yield f"RAG Error: {str(e)}"

    def get_retrieval_context(self, chat_id: str, query: str) -> str:
        """Return the assembled retrieval context for diagnostics/evaluation."""
        nodes = self._retrieve_nodes(chat_id, query)
        return self.deps.build_context_fn(nodes)

    def _retrieve_nodes(self, chat_id: str, query: str):
        """Load stored index, retrieve candidate nodes, and apply reranking."""
        embed_model = self.deps.ensure_embedding_model_fn()
        persist_dir = self.deps.index_dir_fn(chat_id)
        if not os.path.exists(persist_dir):
            raise RuntimeError("Document index not found.")

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(
            similarity_top_k=self.deps.rag_top_k,
            embed_model=embed_model,
        )
        retrieved_nodes = retriever.retrieve(query)
        return self.deps.rerank_nodes_fn(query, retrieved_nodes)


_DEFAULT_ENGINE = RAGEngine()


def ingest_document(file_path: str, chat_id: str):
    """Module-level wrapper that ingests one uploaded document."""
    return _DEFAULT_ENGINE.ingest_document(file_path=file_path, chat_id=chat_id)


def stream_rag_response(
    chat_id: str,
    query: str,
    model: str,
    temperature: float,
    history: List[dict] | None = None,
):
    """Module-level wrapper that streams RAG answer text."""
    yield from _DEFAULT_ENGINE.stream_rag_response(
        chat_id=chat_id,
        query=query,
        model=model,
        temperature=temperature,
        history=history,
    )


def get_retrieval_context(chat_id: str, query: str) -> str:
    """Module-level wrapper that returns retrieval context for a query."""
    return _DEFAULT_ENGINE.get_retrieval_context(chat_id=chat_id, query=query)
