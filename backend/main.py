"""FastAPI application entrypoint for upload, ingestion, chat, and log APIs."""

import shutil
import uuid
import os
import logging
import traceback
import threading
import time
from datetime import timedelta
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from .config import settings
from .database import (
    get_db,
    engine,
    Base,
    ensure_chat_turn_metric_columns,
)
from .auth import (
    create_access_token, 
    authenticate_user, 
    get_current_user, 
    init_user_db,
    get_user_by_username,
    create_user,
)
from .rag_engine import get_retrieval_context, ingest_document
from .manager import generate_response_stream
from .model_catalog import get_model_catalog, get_model_config
from .evaluation import evaluate_metrics as _evaluate_metrics
from .main_helpers import (
    build_logs_recent as _build_logs_recent,
    build_logs_summary as _build_logs_summary,
    get_chat_messages as _get_chat_messages,
    create_chat_turn as _create_chat_turn,
    create_upload_session as _create_upload_session,
    estimate_tokens as _estimate_tokens,
    finalize_chat_turn as _finalize_chat_turn,
    get_recent_history as _get_recent_history,
    get_upload_session as _get_upload_session,
    set_ingestion_status as _set_ingestion_status,
)
from .logging_utils import configure_app_file_logging, log_user_event

SUPPORTED_FILE_TYPES = {
    ".pdf": "doc",
    ".doc": "doc",
    ".docx": "doc",
    ".ppt": "doc",
    ".pptx": "doc",
    ".txt": "doc",
    ".csv": "doc",
    ".xls": "doc",
    ".xlsx": "doc",
    ".png": "doc",
    ".jpg": "doc",
    ".jpeg": "doc",
    ".db": "sql",
    ".sqlite": "sql",
    ".sqlite3": "sql",
}

app = FastAPI(
    title="AI Document Chatbot",
    description="LLM Chatbot for your documents and databases. Upload a file, ask questions, and get answers with context-aware intelligence. Supports PDFs, Word docs, images (with OCR), CSVs, Excel files, and SQL databases. Built with FastAPI, LangChain, and Tesseract OCR.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
ensure_chat_turn_metric_columns()
init_user_db()

logging.basicConfig(level=logging.INFO)
logger = configure_app_file_logging(settings.LOG_DIR, logger_name="main")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
INGESTION_STATUS = {}


def _write_user_log(user_id: str, event: str, level: int = logging.INFO, **fields):
    """Write an event to the user-specific log file without failing request flow."""
    try:
        log_user_event(
            user_id=user_id,
            log_dir=settings.LOG_DIR,
            event=event,
            level=level,
            **fields,
        )
    except Exception as e:
        logger.warning("Failed to write user log for user=%s event=%s err=%s", user_id, event, e)


def _run_ingestion(chat_id: str, owner: str, file_path: str):
    """Run background document ingestion and persist ingestion state transitions."""
    _write_user_log(owner, "ingestion_started", chat_id=chat_id, file_path=file_path)
    try:
        _set_ingestion_status(INGESTION_STATUS, chat_id, owner, "processing")
        ingest_document(file_path, chat_id)
        _set_ingestion_status(INGESTION_STATUS, chat_id, owner, "ready")
        logger.info("Ingestion complete for chat_id=%s", chat_id)
        _write_user_log(owner, "ingestion_completed", chat_id=chat_id, state="ready")
    except Exception as e:
        _set_ingestion_status(
            INGESTION_STATUS,
            chat_id,
            owner,
            "failed",
            f"{e}\n{traceback.format_exc()}",
        )
        logger.exception("Ingestion failed for chat_id=%s: %s", chat_id, e)
        _write_user_log(
            owner,
            "ingestion_failed",
            level=logging.ERROR,
            chat_id=chat_id,
            error=str(e),
        )


def _ensure_supported_extension(ext: str):
    """Validate uploaded file extension against supported file types."""
    if ext not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use /supported-files for allowed extensions.",
        )


def _register_upload_session(
    chat_id: str,
    user_id: str,
    filename: str,
    save_path: str,
    ext: str,
) -> str:
    """Create upload session metadata and start ingestion for document-like files."""
    file_type = SUPPORTED_FILE_TYPES[ext]
    state = "ready" if file_type == "sql" else "processing"

    _create_upload_session(
        chat_id=chat_id,
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_path=save_path,
        state=state,
    )
    _set_ingestion_status(INGESTION_STATUS, chat_id, user_id, state)
    _write_user_log(
        user_id,
        "upload_session_created",
        chat_id=chat_id,
        filename=filename,
        file_type=file_type,
        state=state,
    )

    if file_type == "doc":
        logger.info("Detected Document/Image: %s. Starting Background Ingestion...", filename)
        thread = threading.Thread(
            target=_run_ingestion,
            args=(chat_id, user_id, save_path),
            daemon=True,
        )
        thread.start()
    else:
        logger.info("Detected SQL Database: %s", filename)

    return file_type


def _resolve_session_for_chat(chat_id: str, current_user: str, file_type: str, filename: str):
    """Resolve and authorize chat session metadata before answering a query."""
    session = _get_upload_session(chat_id)
    if session:
        if session.user_id != current_user:
            raise HTTPException(status_code=403, detail="Not authorized for this chat session.")
        if session.status == "processing":
            raise HTTPException(status_code=409, detail="Indexing is still in progress. Please wait.")
        if session.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Indexing failed: {session.last_error or 'unknown error'}",
            )
        return session.file_type, session.filename, session.file_path

    # Backward compatibility for older sessions.
    file_path = os.path.join(settings.UPLOAD_DIR, f"{chat_id}_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File session not found.")
    if file_type == "doc" and not os.path.exists(os.path.join(settings.CHROMA_PATH, chat_id)):
        raise HTTPException(status_code=409, detail="Document index not ready yet. Please wait.")
    return file_type, filename, file_path


def _validate_generation_request(model: str, temperature: float):
    """Validate model choice and temperature bounds for generation request."""
    if not 0.0 <= temperature <= 2.0:
        raise HTTPException(status_code=422, detail="temperature must be between 0.0 and 2.0")
    try:
        get_model_config(model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _stream_chunk_to_text(chunk) -> str:
    """Normalize heterogeneous model chunk payloads into plain text."""
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="ignore")
    if isinstance(chunk, list):
        parts: list[str] = []
        for item in chunk:
            text = _stream_chunk_to_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if isinstance(chunk, dict):
        for key in ("text", "content"):
            value = chunk.get(key)
            if value is not None:
                return _stream_chunk_to_text(value)
        return ""
    content = getattr(chunk, "content", None)
    if content is not None:
        return _stream_chunk_to_text(content)
    return str(chunk)

class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=8, max_length=256)


class UserCreateResponse(BaseModel):
    id: int
    username: str

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    """Authenticate user credentials and issue a bearer access token."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning("Login failed for username=%s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    logger.info(f"User '{user.username}' logged in successfully.")
    _write_user_log(
        user.username,
        "login_success",
        token_ttl_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users", response_model=UserCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_user_api(
    payload: UserCreateRequest,
    db: Session = Depends(get_db),
):
    """Create a new user account when username is not already taken."""
    existing_user = get_user_by_username(db, payload.username)
    if existing_user:
        logger.warning("Create user failed, username exists: %s", payload.username)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists.",
        )

    user = create_user(db, payload.username, payload.password)
    logger.info(f"User '{user.username}' created.")
    _write_user_log(user.username, "user_created")
    return UserCreateResponse(id=user.id, username=user.username)

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """Store uploaded file, register session, and trigger ingestion when required."""
    chat_id = str(uuid.uuid4())
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]
    save_path = os.path.join(settings.UPLOAD_DIR, f"{chat_id}_{filename}")
    _ensure_supported_extension(ext)
    _write_user_log(current_user, "upload_requested", chat_id=chat_id, filename=filename, ext=ext)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {save_path}")
        file_type = _register_upload_session(
            chat_id=chat_id,
            user_id=current_user,
            filename=filename,
            save_path=save_path,
            ext=ext,
        )
        _write_user_log(
            current_user,
            "upload_stored",
            chat_id=chat_id,
            filename=filename,
            file_type=file_type,
            ingestion_state=INGESTION_STATUS[chat_id]["state"],
        )

        return {
            "status": "success",
            "chat_id": chat_id,
            "filename": filename,
            "file_type": file_type,
            "ingestion_state": INGESTION_STATUS[chat_id]["state"],
            "message": (
                "File uploaded. Indexing started in background."
                if file_type == "doc"
                else "File processed and ready for chat."
            ),
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        _write_user_log(
            current_user,
            "upload_failed",
            level=logging.ERROR,
            chat_id=chat_id,
            filename=filename,
            error=str(e),
        )
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/ingestion-status/{chat_id}")
async def ingestion_status(chat_id: str, current_user: str = Depends(get_current_user)):
    """Return ingestion status for a chat session if the caller owns that session."""
    # Lets the UI poll background indexing state so chat only starts when the index is ready.
    status_info = INGESTION_STATUS.get(chat_id)
    if not status_info:
        session = _get_upload_session(chat_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown chat_id.")
        if session.user_id != current_user:
            raise HTTPException(status_code=403, detail="Not authorized for this chat session.")
        return {
            "chat_id": chat_id,
            "state": session.status,
            "error": session.last_error,
        }
    if status_info.get("owner") != current_user:
        raise HTTPException(status_code=403, detail="Not authorized for this chat session.")
    return {
        "chat_id": chat_id,
        "state": status_info["state"],
        "error": status_info.get("error"),
    }


@app.get("/chat")
async def chat_stream(
    chat_id: str,
    query: str,
    file_type: str,
    filename: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    current_user: str = Depends(get_current_user),
):
    """Stream chat response for the selected session/model and persist turn metrics."""
    file_type, filename, file_path = _resolve_session_for_chat(
        chat_id=chat_id,
        current_user=current_user,
        file_type=file_type,
        filename=filename,
    )
    _validate_generation_request(model=model, temperature=temperature)

    history = _get_recent_history(
        chat_id=chat_id,
        user_id=current_user,
        limit=settings.MEMORY_MAX_TURNS,
    )
    turn_id = _create_chat_turn(
        user_id=current_user,
        chat_id=chat_id,
        query=query,
        model_used=model,
    )
    _write_user_log(
        current_user,
        "chat_turn_started",
        chat_id=chat_id,
        turn_id=turn_id,
        model=model,
        query_chars=len(query),
    )

    async def stream_and_record():
        """Stream tokens to client and finalize DB logs after generation completes."""
        started_at = time.perf_counter()
        chunks: list[str] = []
        stream_error: str | None = None
        try:
            async for chunk in generate_response_stream(
                chat_id,
                query,
                file_type,
                file_path,
                model,
                temperature,
                history=history,
            ):
                text_chunk = _stream_chunk_to_text(chunk)
                if text_chunk:
                    chunks.append(text_chunk)
                    yield text_chunk
        except Exception as e:
            stream_error = str(e)
            err_msg = f"Generation Error: {stream_error}"
            chunks.append(err_msg)
            yield err_msg
        finally:
            response_text = "".join(chunks)
            latency_ms = max(1, int((time.perf_counter() - started_at) * 1000))
            input_tokens = _estimate_tokens(query)
            output_tokens = _estimate_tokens(response_text)
            total_tokens = input_tokens + output_tokens
            context_text = response_text if file_type == "sql" else ""
            if file_type == "doc":
                try:
                    context_text = get_retrieval_context(chat_id=chat_id, query=query)
                except Exception:
                    context_text = ""
            metrics = _evaluate_metrics(
                query=query,
                response=response_text,
                context=context_text,
                error_text=stream_error,
            )
            evaluation_score = metrics["overall"]
            _finalize_chat_turn(
                turn_id=turn_id,
                response_text=response_text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                evaluation_score=evaluation_score,
                answer_relevancy=metrics["answer_relevancy"],
                faithfulness_score=metrics["faithfulness"],
                contextual_precision=metrics["contextual_precision"],
                error_text=stream_error,
            )
            _write_user_log(
                current_user,
                "chat_turn_completed",
                chat_id=chat_id,
                turn_id=turn_id,
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                score=round(float(evaluation_score), 4),
                had_error=bool(stream_error),
            )

    return StreamingResponse(stream_and_record(), media_type="text/plain")


@app.get("/logs/summary")
def logs_summary(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return aggregated upload/chat metrics for the authenticated user."""
    return _build_logs_summary(db=db, current_user=current_user)


@app.get("/logs/recent")
def logs_recent(
    limit: int = 30,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return recent upload and chat records for the authenticated user."""
    # Returns recent upload/chat events for the signed-in user to power dashboard tables.
    return _build_logs_recent(db=db, current_user=current_user, limit=limit)


@app.get("/chat-history/{chat_id}")
def chat_history(
    chat_id: str,
    limit: int = 100,
    current_user: str = Depends(get_current_user),
):
    """Return persisted chat messages for a session owned by the current user."""
    session = _get_upload_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown chat_id.")
    if session.user_id != current_user:
        raise HTTPException(status_code=403, detail="Not authorized for this chat session.")
    safe_limit = max(1, min(limit, 300))
    return {
        "chat_id": chat_id,
        "filename": session.filename,
        "file_type": session.file_type,
        "messages": _get_chat_messages(chat_id=chat_id, user_id=current_user, limit=safe_limit),
    }


@app.get("/models")
def get_models():
    """Return available model catalog with provider readiness flags."""
    return {"models": get_model_catalog()}

@app.get("/supported-files")
def supported_files():
    """Return supported upload extensions and parser behavior notes."""
    return {
        "supported_extensions": sorted(SUPPORTED_FILE_TYPES.keys()),
        "sql_extensions": [ext for ext, kind in SUPPORTED_FILE_TYPES.items() if kind == "sql"],
        "rag_extensions": [ext for ext, kind in SUPPORTED_FILE_TYPES.items() if kind == "doc"],
        "notes": [
            "Images and scanned PDFs use explicit OCR via Tesseract during ingestion.",
            "Legacy .doc may require extra system converters depending on the host.",
            "PPTX is parsed slide-wise; legacy .ppt uses best-effort reader fallback.",
            "CSV/XLS/XLSX are chunked by rows before indexing to avoid token-limit failures.",
        ],
    }

@app.get("/health")
def health_check():
    """Return a minimal health payload for uptime monitoring."""
    return {"status": "active", "service": "AI-Doc-Chat-v1.0"}
