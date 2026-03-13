"""Embedding model loaders used by the RAG indexing and retrieval pipeline."""

import warnings

from llama_index.core import Settings as LlamaSettings

from .config import settings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Call to deprecated class HuggingFaceInferenceAPIEmbedding.*",
)


def _build_safe_sync_hf_embedding(model_name: str, token: str, timeout: float = 60.0):
    """Build a sync-safe HuggingFace API embedding client with compatibility fallbacks."""
    try:
        from llama_index.embeddings.huggingface_api import (  # type: ignore
            HuggingFaceInferenceAPIEmbedding,
        )
        embedding_utils_module = "llama_index.embeddings.huggingface_api.utils"
    except Exception as e:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                from llama_index.embeddings.huggingface import (  # type: ignore
                    HuggingFaceInferenceAPIEmbedding,
                )
                embedding_utils_module = "llama_index.embeddings.huggingface.utils"
            except Exception:
                raise RuntimeError(
                    "Missing HuggingFace embedding integrations. Install either "
                    "`llama-index-embeddings-huggingface-api` (preferred) or "
                    "`llama-index-embeddings-huggingface`."
                ) from e

    format_query = None
    format_text = None
    try:
        if embedding_utils_module == "llama_index.embeddings.huggingface_api.utils":
            from llama_index.embeddings.huggingface_api.utils import (  # type: ignore
                format_query as _format_query,
                format_text as _format_text,
            )
        else:
            from llama_index.embeddings.huggingface.utils import (  # type: ignore
                format_query as _format_query,
                format_text as _format_text,
            )
        format_query = _format_query
        format_text = _format_text
    except Exception:
        pass

    def _fmt_query(value: str, model: str, instruction):
        """Format query text with model-specific instruction when formatter is available."""
        if format_query is not None:
            return format_query(value, model, instruction)
        return value

    def _fmt_text(value: str, model: str, instruction):
        """Format document text with model-specific instruction when formatter is available."""
        if format_text is not None:
            return format_text(value, model, instruction)
        return value

    class _SafeSyncHuggingFaceInferenceAPIEmbedding(HuggingFaceInferenceAPIEmbedding):
        def _embed_sync(self, text: str):
            """Run sync embedding call and normalize response shape into a flat list."""
            embedding = self._sync_client.feature_extraction(text)
            if not hasattr(embedding, "shape"):
                return list(embedding)
            if len(embedding.shape) == 1:
                return embedding.tolist()

            embedding = embedding.squeeze(axis=0)
            if len(embedding.shape) == 1:
                return embedding.tolist()
            if self.pooling is None:
                raise ValueError(
                    f"Pooling is required for {self.model_name} because it returned >1-D output."
                )
            return self.pooling(embedding).tolist()

        def _get_query_embedding(self, query: str):
            """Embed a query using sync transport to avoid async event-loop issues."""
            return self._embed_sync(
                _fmt_query(query, self.model_name, self.query_instruction)
            )

        def _get_text_embedding(self, text: str):
            """Embed one text chunk using sync transport."""
            return self._embed_sync(
                _fmt_text(text, self.model_name, self.text_instruction)
            )

        def _get_text_embeddings(self, texts):
            """Embed multiple text chunks using repeated sync calls."""
            return [self._get_text_embedding(text) for text in texts]

        def get_text_embedding_batch(self, texts, show_progress=False, **kwargs):
            """Override batch embedding to keep execution fully synchronous."""
            # Bypass async internals and run sync calls only.
            return [self._get_text_embedding(t) for t in texts]

        async def _aget_query_embedding(self, query: str):
            """Async compatibility wrapper that delegates to sync query embedding."""
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str):
            """Async compatibility wrapper that delegates to sync text embedding."""
            return self._get_text_embedding(text)

        async def _aget_text_embeddings(self, texts):
            """Async compatibility wrapper that delegates to sync batch embedding."""
            return self._get_text_embeddings(texts)

    return _SafeSyncHuggingFaceInferenceAPIEmbedding(
        model_name=model_name,
        token=token,
        timeout=timeout,
        embed_batch_size=4,
    )


def _load_hf_embedding_remote():
    """Load HuggingFace remote embedding model with ordered fallbacks."""
    if not settings.HF_TOKEN:
        raise ValueError(
            "HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is required when EMBEDDING_PROVIDER=huggingface."
        )

    last_error = None
    for model_name in (
        settings.EMBEDDING_MODEL_NAME,
        settings.EMBEDDING_FALLBACK_MODEL_NAME,
        settings.EMBEDDING_LAST_RESORT_MODEL_NAME,
    ):
        try:
            return _build_safe_sync_hf_embedding(
                model_name=model_name,
                token=settings.HF_TOKEN,
                timeout=60.0,
            )
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Unable to initialize HF embedding model: {last_error}")


def _load_gemini_embedding():
    """Load Gemini embedding model with fallback model id."""
    try:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    except Exception as e:
        raise RuntimeError(
            "Gemini embedding package is missing. Install `llama-index-embeddings-google-genai`."
        ) from e

    if not settings.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is required when EMBEDDING_PROVIDER=gemini."
        )

    try:
        return GoogleGenAIEmbedding(
            model_name=settings.GEMINI_EMBEDDING_MODEL_NAME,
            api_key=settings.GOOGLE_API_KEY,
            embed_batch_size=8,
        )
    except Exception:
        return GoogleGenAIEmbedding(
            model_name=settings.GEMINI_EMBEDDING_FALLBACK_MODEL_NAME,
            api_key=settings.GOOGLE_API_KEY,
            embed_batch_size=8,
        )


def ensure_embedding_model():
    """Resolve configured embedding provider and bind it into LlamaIndex settings."""
    provider = settings.EMBEDDING_PROVIDER.strip().lower()
    if provider == "gemini":
        embed_model = _load_gemini_embedding()
    elif provider == "huggingface":
        embed_model = _load_hf_embedding_remote()
    else:
        raise ValueError("Unsupported EMBEDDING_PROVIDER. Use 'huggingface' or 'gemini'.")

    LlamaSettings.embed_model = embed_model
    return embed_model
