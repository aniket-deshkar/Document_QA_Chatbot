"""Database models and session utilities for users, uploads, and chat history."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

engine = create_engine(
    settings.SYSTEM_DB_PATH,
    connect_args={"check_same_thread": False}
    if settings.SYSTEM_DB_PATH.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, index=True, nullable=False)
    password_hash = Column(String(512), nullable=False)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), index=True, nullable=False)
    chat_id = Column(String(128), index=True, nullable=False)
    query = Column(Text, nullable=False)
    model_used = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UploadSession(Base):
    __tablename__ = "upload_sessions"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String(128), unique=True, index=True, nullable=False)
    user_id = Column(String(128), index=True, nullable=False)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_path = Column(String(1024), nullable=False)
    status = Column(String(32), nullable=False, default="processing")
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), index=True, nullable=False)
    chat_id = Column(String(128), index=True, nullable=False)
    model_used = Column(String(128), nullable=False)
    query = Column(Text, nullable=False)
    response_text = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    evaluation_score = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    faithfulness_score = Column(Float, nullable=True)
    contextual_precision = Column(Float, nullable=True)
    error_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_db():
    """Provide a request-scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_chat_turn_metric_columns():
    """Backfill missing metric columns in SQLite chat_turns table for compatibility."""
    if not settings.SYSTEM_DB_PATH.startswith("sqlite"):
        return
    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(chat_turns);")
        existing = {row[1] for row in cursor.fetchall()}
        missing_columns = {
            "answer_relevancy": "REAL",
            "faithfulness_score": "REAL",
            "contextual_precision": "REAL",
        }
        for name, col_type in missing_columns.items():
            if name not in existing:
                cursor.execute(f"ALTER TABLE chat_turns ADD COLUMN {name} {col_type};")
        conn.commit()
    finally:
        conn.close()
