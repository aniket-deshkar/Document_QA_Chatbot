"""Orchestrates the chat path between SQL-agent responses and RAG responses."""

from .rag_engine import stream_rag_response
from .sql_engine import get_sql_agent

async def generate_response_stream(
    chat_id: str,
    query: str,
    file_type: str,
    file_path: str,
    model: str,
    temperature: float,
    history: list[dict] | None = None,
):
    """Route a chat request to SQL agent or RAG engine and stream back output."""
    
    # --- PATH A: SQL Database ---
    if file_type == "sql":
        agent = get_sql_agent(file_path, model=model, temperature=temperature)
        # SQL Agents are "Action-Observation" loops, so they don't stream token-by-token well.
        # We run it and yield the final result.
        try:
            if history:
                turns = []
                for turn in history:
                    u = (turn.get("user") or "").strip()
                    a = (turn.get("assistant") or "").strip()
                    if u:
                        turns.append(f"User: {u}")
                    if a:
                        turns.append(f"Assistant: {a}")
                history_text = "\n".join(turns)
                effective_query = (
                    "Use the following conversation context for follow-up resolution.\n"
                    f"{history_text}\n\nCurrent user question:\n{query}"
                )
            else:
                effective_query = query
            response = agent.run(effective_query)
            yield response
        except Exception as e:
            yield f"SQL Error: {str(e)}"

    # --- PATH B: Document / Image (OCR) ---
    else:
        for token in stream_rag_response(
            chat_id=chat_id,
            query=query,
            model=model,
            temperature=temperature,
            history=history or [],
        ):
            yield token
