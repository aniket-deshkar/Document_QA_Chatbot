"""Model registry and provider-specific LangChain client builders."""

from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from .config import settings


# Model list tuned for "latest + free-tier friendly" options.
# Free-tier quotas and exact availability change frequently by provider/account.
MODEL_CATALOG: List[Dict[str, str]] = [
    # Google AI Studio (Gemini)
    {
        "id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "provider": "google",
        "provider_label": "Google AI Studio",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Local OCR (Tesseract) + LLM reasoning",
        "free_tier": "yes (AI Studio quota-based)",
        "use_for": "Best default Gemini option for speed + quality.",
        "details": "Strong multimodal model for document/image grounded chat.",
    },
    {
        "id": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "provider": "google",
        "provider_label": "Google AI Studio",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Local OCR (Tesseract) + LLM reasoning",
        "free_tier": "limited/varies by project",
        "use_for": "High-quality reasoning for complex multi-step queries.",
        "details": "Use when answer quality is more important than latency.",
    },
    {
        "id": "gemini-flash-latest",
        "label": "Gemini Flash Latest",
        "provider": "google",
        "provider_label": "Google AI Studio",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Local OCR (Tesseract) + LLM reasoning",
        "free_tier": "yes (AI Studio quota-based)",
        "use_for": "Auto-track latest Flash alias without manual updates.",
        "details": "Useful when you want latest Flash improvements quickly.",
    },
    {
        "id": "gemini-2.5-flash-lite-preview-09-2025",
        "label": "Gemini 2.5 Flash-Lite (Preview)",
        "provider": "google",
        "provider_label": "Google AI Studio",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Local OCR (Tesseract) + LLM reasoning",
        "free_tier": "yes (preview quota-based)",
        "use_for": "Lowest-cost Gemini option for light QA and extraction.",
        "details": "Great for high-throughput, lower-complexity tasks.",
    },
    # Groq
    {
        "id": "openai/gpt-oss-120b",
        "label": "GPT OSS 120B",
        "provider": "groq",
        "provider_label": "Groq",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (Groq developer quota)",
        "use_for": "Latest high-quality open-text reasoning on Groq.",
        "details": "Large model for deeper reasoning and tool-heavy prompts.",
    },
    {
        "id": "openai/gpt-oss-20b",
        "label": "GPT OSS 20B",
        "provider": "groq",
        "provider_label": "Groq",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (Groq developer quota)",
        "use_for": "Fast and economical open model for general chat + RAG.",
        "details": "Lower latency than larger OSS models.",
    },
    {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "label": "Llama 4 Scout 17B",
        "provider": "groq",
        "provider_label": "Groq",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Local OCR preferred for stable extraction",
        "free_tier": "yes (Groq developer quota)",
        "use_for": "Multimodal reasoning where image understanding helps.",
        "details": "Use when you need vision-capable open model behavior.",
    },
    {
        "id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "label": "Llama 4 Maverick 17B",
        "provider": "groq",
        "provider_label": "Groq",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Local OCR preferred for stable extraction",
        "free_tier": "yes (Groq developer quota)",
        "use_for": "Multimodal option with stronger broad reasoning profile.",
        "details": "Useful for mixed image + long context tasks.",
    },
    {
        "id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B Versatile",
        "provider": "groq",
        "provider_label": "Groq",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (Groq developer quota)",
        "use_for": "Strong general-purpose RAG and reasoning.",
        "details": "Stable fallback for broad text-heavy workloads.",
    },
    {
        "id": "qwen/qwen3-32b",
        "label": "Qwen3 32B",
        "provider": "groq",
        "provider_label": "Groq",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (Groq developer quota)",
        "use_for": "High-quality text reasoning and coding-style prompts.",
        "details": "Good alternative to Llama-family prompts.",
    },
    # Hugging Face Inference (OpenAI-compatible router)
    {
        "id": "zai-org/GLM-4.5",
        "label": "HF GLM-4.5",
        "provider": "huggingface",
        "provider_label": "Hugging Face Inference",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (monthly credits)",
        "use_for": "Latest high-capability text model via HF routed API.",
        "details": "Uses HF free monthly credits before pay-as-you-go.",
    },
    {
        "id": "zai-org/GLM-4.5V",
        "label": "HF GLM-4.5V",
        "provider": "huggingface",
        "provider_label": "Hugging Face Inference",
        "category": "multimodal",
        "supports_vision": "yes",
        "ocr_mode": "Vision-capable + local OCR fallback",
        "free_tier": "yes (monthly credits)",
        "use_for": "Vision + text prompts through HF provider routing.",
        "details": "Useful when image-grounded responses are needed.",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1:fastest",
        "label": "HF DeepSeek-R1 (Fastest Route)",
        "provider": "huggingface",
        "provider_label": "Hugging Face Inference",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (monthly credits)",
        "use_for": "Reasoning-heavy tasks where chain-of-thought style helps.",
        "details": "Provider-routed variant optimized for latency.",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "label": "HF Llama 3.3 70B Instruct",
        "provider": "huggingface",
        "provider_label": "Hugging Face Inference",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (monthly credits)",
        "use_for": "General high-quality instruction and RAG answers.",
        "details": "Strong open model fallback on HF routing.",
    },
    {
        "id": "Qwen/Qwen3-32B",
        "label": "HF Qwen3 32B",
        "provider": "huggingface",
        "provider_label": "Hugging Face Inference",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes (monthly credits)",
        "use_for": "Good multilingual/text reasoning alternative.",
        "details": "Useful when Qwen prompt behavior fits your use case.",
    },
    # OpenRouter
    {
        "id": "openrouter/free",
        "label": "OpenRouter Free Router",
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "category": "mixed",
        "supports_vision": "varies",
        "ocr_mode": "Depends on routed model; local OCR always available",
        "free_tier": "yes (free-routed)",
        "use_for": "Automatically picks currently available free models.",
        "details": "Best if you want hands-off free model rotation.",
    },
    {
        "id": "openai/gpt-oss-20b:free",
        "label": "OpenRouter GPT OSS 20B (Free)",
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes",
        "use_for": "Free reasoning-oriented text model on OpenRouter.",
        "details": "Good fallback when you want deterministic free routing.",
    },
    {
        "id": "meta-llama/llama-3.2-3b-instruct:free",
        "label": "OpenRouter Llama 3.2 3B (Free)",
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes",
        "use_for": "Very low-cost baseline for simple extraction/QA.",
        "details": "Lightweight model for quick, cheap iterations.",
    },
    {
        "id": "google/gemma-3-27b-it:free",
        "label": "OpenRouter Gemma 3 27B (Free)",
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes",
        "use_for": "Balanced free model for richer answers than tiny models.",
        "details": "Good middle ground between speed and response quality.",
    },
    {
        "id": "qwen/qwen3-235b-a22b:free",
        "label": "OpenRouter Qwen3 235B A22B (Free)",
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "category": "text",
        "supports_vision": "no",
        "ocr_mode": "Local OCR required",
        "free_tier": "yes",
        "use_for": "High-capacity free reasoning when available.",
        "details": "Availability can fluctuate due to free-pool demand.",
    },
]


def _provider_enabled(provider: str) -> bool:
    """Return whether required API key for the provider is configured."""
    return (
        (provider == "google" and bool(settings.GOOGLE_API_KEY))
        or (provider == "groq" and bool(settings.GROQ_API_KEY))
        or (provider == "openrouter" and bool(settings.OPENROUTER_API_KEY))
        or (provider == "huggingface" and bool(settings.HF_TOKEN))
    )


def get_model_catalog() -> List[Dict[str, str]]:
    """Return model metadata with runtime enabled/disabled status per provider."""
    models = []
    for model in MODEL_CATALOG:
        models.append({**model, "enabled": _provider_enabled(model["provider"])})
    return models


def get_model_config(model_id: str) -> Dict[str, str]:
    """Fetch model configuration by model id."""
    for model in MODEL_CATALOG:
        if model["id"] == model_id:
            return model
    raise ValueError(f"Unsupported model: {model_id}")


def build_langchain_llm(model_id: str, temperature: float):
    """Build a provider-specific LangChain chat client for the selected model."""
    model = get_model_config(model_id)
    provider = model["provider"]

    if provider == "google":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured.")
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")
        return ChatGroq(
            model=model_id,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )

    if provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
        return ChatOpenAI(
            model=model_id,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    if provider == "huggingface":
        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is not configured.")
        return ChatOpenAI(
            model=model_id,
            api_key=settings.HF_TOKEN,
            base_url=settings.HUGGINGFACE_BASE_URL,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider}")
