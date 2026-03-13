"""Prompt templates and message builders for retrieval-grounded chat."""

from typing import List

from .config import settings

SYSTEM_PROMPT = (
    "You are a document QA assistant.\n"
    "Rules:\n"
    "1) For document facts, use only the retrieved document context.\n"
    "2) For conversation-meta questions (for example: 'what was my last question?'), use chat history.\n"
    "3) If answer is missing from context/history, reply exactly: I do not know based on the uploaded document.\n"
    "4) For analytical questions, provide concise step-by-step reasoning and final answer.\n"
    "5) You may use analogies for explanation style, but do not add new facts outside context/history.\n"
    "6) Do not give vague/general filler; be specific and concise."
)


def build_history_messages(history: List[dict]) -> List[tuple]:
    """Convert stored chat turns into LangChain-compatible message tuples."""
    messages: List[tuple] = []
    for turn in history:
        user_text = (turn.get("user") or "").strip()
        assistant_text = (turn.get("assistant") or "").strip()
        if user_text:
            messages.append(("human", user_text[: settings.MEMORY_MAX_CHARS_PER_TURN]))
        if assistant_text:
            messages.append(("assistant", assistant_text[: settings.MEMORY_MAX_CHARS_PER_TURN]))
    return messages


def _render_history_block(history: List[dict] | None) -> str:
    """Render recent conversation history as plain text for prompt context."""
    if not history:
        return "No prior conversation turns."
    lines: List[str] = []
    for turn in history[-settings.MEMORY_MAX_TURNS :]:
        user_text = (turn.get("user") or "").strip()
        assistant_text = (turn.get("assistant") or "").strip()
        if user_text:
            lines.append(f"User: {user_text[: settings.MEMORY_MAX_CHARS_PER_TURN]}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text[: settings.MEMORY_MAX_CHARS_PER_TURN]}")
    return "\n".join(lines) if lines else "No prior conversation turns."


def build_rag_messages(context: str, query: str, history: List[dict] | None = None) -> List[tuple]:
    """Build final prompt message list combining rules, history, and retrieved context."""
    messages = [("system", SYSTEM_PROMPT)]
    if history:
        messages.extend(build_history_messages(history))
    history_block = _render_history_block(history)
    messages.append(
        (
            "human",
            f"Conversation history:\n{history_block}\n\n"
            f"Retrieved document context:\n{context}\n\n"
            f"Current question:\n{query}\n\n"
            "Instructions:\n"
            "- If the question refers to prior turns, use conversation history.\n"
            "- Otherwise answer from retrieved document context.\n"
            "- If unavailable, return the exact fallback sentence from the system prompt.",
        )
    )
    return messages
