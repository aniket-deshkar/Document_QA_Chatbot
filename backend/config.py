"""Centralized application settings loaded from environment variables and .env."""

import os
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Settings(BaseSettings):
    SECRET_KEY: str = "962860b5a2246c7f3aa0f56a08261a831539b18b5819ed05ab0bf3bc339d4b68"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SYSTEM_DB_PATH: str = "sqlite:///./system.db"
    GOOGLE_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    GROQ_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY"),
    )
    OPENROUTER_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY"),
    )
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    HF_TOKEN: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    )
    HUGGINGFACE_BASE_URL: str = "https://router.huggingface.co/v1"
    COHERE_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("COHERE_API_KEY"),
    )
    
    UPLOAD_DIR: str = "uploads"
    CHROMA_PATH: str = "chroma_db"
    MODEL_CACHE_PATH: str = ".model_cache"
    LOG_DIR: str = "logs"
    EMBEDDING_PROVIDER: str = "huggingface"
    GEMINI_EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"
    GEMINI_EMBEDDING_FALLBACK_MODEL_NAME: str = "text-embedding-005"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_FALLBACK_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_LAST_RESORT_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_TRUST_REMOTE_CODE: bool = True
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 4
    ENABLE_COHERE_RERANK: bool = True
    COHERE_RERANK_MODEL: str = "rerank-v3.5"
    COHERE_RERANK_TOP_N: int = 3
    RAG_CONTEXT_MAX_CHARS: int = 12000
    TABLE_ROWS_PER_CHUNK: int = 500
    TABLE_MAX_ROWS: int = 3000
    TABLE_TOKEN_CHUNK_SIZE: int = 3000
    OCR_IMAGE_DPI: int = 220
    PDF_TEXT_MIN_LENGTH_FOR_NO_OCR: int = 40
    TESSERACT_CMD: str = ""
    MEMORY_MAX_TURNS: int = 6
    MEMORY_MAX_CHARS_PER_TURN: int = 800
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PATH, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_PATH, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)
