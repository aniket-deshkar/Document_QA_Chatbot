"""Shared helper functions for persistence, history, and user-wise log views."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func

from .database import ChatTurn, SessionLocal, UploadSession


def set_ingestion_status(
    status_store: dict[str, dict[str, Any]],
    chat_id: str,
    owner: str,
    state: str,
    error: str | None = None,
):
    """Update in-memory and DB ingestion status for a chat upload session."""
    status_store[chat_id] = {
        "state": state,
        "owner": owner,
        "error": error,
    }
    db = SessionLocal()
    try:
        session = db.query(UploadSession).filter(UploadSession.chat_id == chat_id).first()
        if session:
            session.status = state
            session.last_error = error
            db.commit()
    finally:
        db.close()


def create_upload_session(
    chat_id: str,
    user_id: str,
    filename: str,
    file_type: str,
    file_path: str,
    state: str,
):
    """Create a persisted upload session record for a user document."""
    db = SessionLocal()
    try:
        db.add(
            UploadSession(
                chat_id=chat_id,
                user_id=user_id,
                filename=filename,
                file_type=file_type,
                file_path=file_path,
                status=state,
                last_error=None,
            )
        )
        db.commit()
    finally:
        db.close()


def get_upload_session(chat_id: str):
    """Fetch upload session metadata by chat id."""
    db = SessionLocal()
    try:
        return db.query(UploadSession).filter(UploadSession.chat_id == chat_id).first()
    finally:
        db.close()


def estimate_tokens(text: str) -> int:
    """Estimate token count with a simple character-based heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_recent_history(chat_id: str, user_id: str, limit: int):
    """Return recent conversation turns for prompt memory injection."""
    db = SessionLocal()
    try:
        turns = (
            db.query(ChatTurn)
            .filter(ChatTurn.chat_id == chat_id, ChatTurn.user_id == user_id)
            .order_by(ChatTurn.created_at.desc())
            .limit(limit)
            .all()
        )
        turns.reverse()
        return [{"user": t.query, "assistant": t.response_text or ""} for t in turns]
    finally:
        db.close()


def get_chat_messages(chat_id: str, user_id: str, limit: int):
    """Return full chat messages in chronological order for UI restoration."""
    db = SessionLocal()
    try:
        turns = (
            db.query(ChatTurn)
            .filter(ChatTurn.chat_id == chat_id, ChatTurn.user_id == user_id)
            .order_by(ChatTurn.created_at.asc())
            .limit(limit)
            .all()
        )
        messages: list[dict[str, str]] = []
        for turn in turns:
            user_text = (turn.query or "").strip()
            assistant_text = (turn.response_text or "").strip()
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        return messages
    finally:
        db.close()


def create_chat_turn(user_id: str, chat_id: str, query: str, model_used: str) -> int:
    """Create an empty chat turn row before streaming response tokens."""
    db = SessionLocal()
    try:
        turn = ChatTurn(
            user_id=user_id,
            chat_id=chat_id,
            model_used=model_used,
            query=query,
        )
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return turn.id
    finally:
        db.close()


def finalize_chat_turn(
    turn_id: int,
    response_text: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    evaluation_score: float,
    answer_relevancy: float | None,
    faithfulness_score: float | None,
    contextual_precision: float | None,
    error_text: str | None,
):
    """Persist final response text, latency, token usage, and evaluation metrics."""
    db = SessionLocal()
    try:
        turn = db.query(ChatTurn).filter(ChatTurn.id == turn_id).first()
        if not turn:
            return
        turn.response_text = response_text
        turn.latency_ms = latency_ms
        turn.input_tokens = input_tokens
        turn.output_tokens = output_tokens
        turn.total_tokens = total_tokens
        turn.evaluation_score = evaluation_score
        turn.answer_relevancy = answer_relevancy
        turn.faithfulness_score = faithfulness_score
        turn.contextual_precision = contextual_precision
        turn.error_text = error_text
        db.commit()
    finally:
        db.close()


def serialize_upload(upload: UploadSession) -> dict[str, Any]:
    """Convert an UploadSession ORM row into API-ready dictionary."""
    return {
        "chat_id": upload.chat_id,
        "filename": upload.filename,
        "file_type": upload.file_type,
        "status": upload.status,
        "last_error": upload.last_error,
        "created_at": upload.created_at.isoformat(),
        "updated_at": upload.updated_at.isoformat() if upload.updated_at else None,
    }


def serialize_turn(turn: ChatTurn) -> dict[str, Any]:
    """Convert a ChatTurn ORM row into API-ready dictionary."""
    return {
        "chat_id": turn.chat_id,
        "model_used": turn.model_used,
        "query": turn.query,
        "response_text": turn.response_text,
        "latency_ms": turn.latency_ms,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "total_tokens": turn.total_tokens,
        "evaluation_score": turn.evaluation_score,
        "answer_relevancy": turn.answer_relevancy,
        "faithfulness_score": turn.faithfulness_score,
        "contextual_precision": turn.contextual_precision,
        "error_text": turn.error_text,
        "created_at": turn.created_at.isoformat(),
    }


def build_logs_summary(db, current_user: str) -> dict[str, Any]:
    """Build aggregated user-wise metrics for uploads and chat quality."""
    total_uploads = (
        db.query(func.count(UploadSession.id))
        .filter(UploadSession.user_id == current_user)
        .scalar()
        or 0
    )
    ready_uploads = (
        db.query(func.count(UploadSession.id))
        .filter(UploadSession.user_id == current_user, UploadSession.status == "ready")
        .scalar()
        or 0
    )
    failed_uploads = (
        db.query(func.count(UploadSession.id))
        .filter(UploadSession.user_id == current_user, UploadSession.status == "failed")
        .scalar()
        or 0
    )
    total_turns = (
        db.query(func.count(ChatTurn.id)).filter(ChatTurn.user_id == current_user).scalar() or 0
    )
    avg_latency = db.query(func.avg(ChatTurn.latency_ms)).filter(
        ChatTurn.user_id == current_user
    ).scalar()
    avg_score = db.query(func.avg(ChatTurn.evaluation_score)).filter(
        ChatTurn.user_id == current_user
    ).scalar()
    avg_answer_relevancy = db.query(func.avg(ChatTurn.answer_relevancy)).filter(
        ChatTurn.user_id == current_user
    ).scalar()
    avg_faithfulness = db.query(func.avg(ChatTurn.faithfulness_score)).filter(
        ChatTurn.user_id == current_user
    ).scalar()
    avg_contextual_precision = db.query(func.avg(ChatTurn.contextual_precision)).filter(
        ChatTurn.user_id == current_user
    ).scalar()
    token_sum = (
        db.query(func.coalesce(func.sum(ChatTurn.total_tokens), 0))
        .filter(ChatTurn.user_id == current_user)
        .scalar()
        or 0
    )
    last_upload = (
        db.query(UploadSession)
        .filter(UploadSession.user_id == current_user)
        .order_by(UploadSession.created_at.desc())
        .first()
    )
    last_turn = (
        db.query(ChatTurn)
        .filter(ChatTurn.user_id == current_user)
        .order_by(ChatTurn.created_at.desc())
        .first()
    )
    return {
        "user_id": current_user,
        "uploads": {
            "total": total_uploads,
            "ready": ready_uploads,
            "failed": failed_uploads,
            "last_at": last_upload.created_at.isoformat() if last_upload else None,
        },
        "chat": {
            "total_turns": total_turns,
            "avg_latency_ms": round(float(avg_latency), 2) if avg_latency is not None else None,
            "avg_evaluation_score": round(float(avg_score), 3) if avg_score is not None else None,
            "avg_answer_relevancy": (
                round(float(avg_answer_relevancy), 3)
                if avg_answer_relevancy is not None
                else None
            ),
            "avg_faithfulness": (
                round(float(avg_faithfulness), 3) if avg_faithfulness is not None else None
            ),
            "avg_contextual_precision": (
                round(float(avg_contextual_precision), 3)
                if avg_contextual_precision is not None
                else None
            ),
            "total_tokens_estimated": int(token_sum),
            "last_turn_at": last_turn.created_at.isoformat() if last_turn else None,
        },
    }


def build_logs_recent(db, current_user: str, limit: int) -> dict[str, Any]:
    """Build recent user-wise upload and chat event lists."""
    safe_limit = max(1, min(limit, 200))
    uploads = (
        db.query(UploadSession)
        .filter(UploadSession.user_id == current_user)
        .order_by(UploadSession.created_at.desc())
        .limit(safe_limit)
        .all()
    )
    turns = (
        db.query(ChatTurn)
        .filter(ChatTurn.user_id == current_user)
        .order_by(ChatTurn.created_at.desc())
        .limit(safe_limit)
        .all()
    )
    return {
        "user_id": current_user,
        "uploads": [serialize_upload(upload) for upload in uploads],
        "turns": [serialize_turn(turn) for turn in turns],
    }
