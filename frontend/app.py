"""Streamlit frontend for authentication, file processing, and chat interactions."""

import streamlit as st
import requests
import time
from requests.exceptions import ChunkedEncodingError, RequestException, Timeout

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Universal Doc Chat", layout="wide")

if "chat_id" not in st.session_state: st.session_state.chat_id = None
if "messages" not in st.session_state: st.session_state.messages = []
if "file_meta" not in st.session_state: st.session_state.file_meta = {}
if "access_token" not in st.session_state: st.session_state.access_token = ""
if "auth_user" not in st.session_state: st.session_state.auth_user = ""


def auth_headers():
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def wait_for_ingestion(chat_id: str, timeout_seconds: int = 900, poll_interval: float = 5.0):
    start = time.time()
    while (time.time() - start) < timeout_seconds:
        resp = requests.get(
            f"{API_URL}/ingestion-status/{chat_id}",
            headers=auth_headers(),
            timeout=15,
        )
        if resp.status_code == 401:
            return {
                "state": "failed",
                "error": "Authentication token expired during indexing. Please login again and retry.",
            }
        if resp.status_code != 200:
            return {"state": "failed", "error": resp.text}
        data = resp.json()
        state = data.get("state")
        if state in {"ready", "failed"}:
            return data
        time.sleep(poll_interval)
    return {"state": "failed", "error": f"Indexing timed out after {timeout_seconds} seconds."}


@st.cache_data(show_spinner=False)
def fetch_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return []


def fetch_logs_summary():
    last_error = None
    for _ in range(2):
        try:
            resp = requests.get(
                f"{API_URL}/logs/summary",
                headers=auth_headers(),
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}
        except Timeout:
            last_error = "Backend is busy; logs request timed out."
            time.sleep(0.5)
        except RequestException as e:
            last_error = str(e)
            break
    return {"error": last_error or "Unable to fetch logs summary."}


def fetch_recent_logs(limit: int = 20):
    last_error = None
    for _ in range(2):
        try:
            resp = requests.get(
                f"{API_URL}/logs/recent",
                params={"limit": limit},
                headers=auth_headers(),
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}
        except Timeout:
            last_error = "Backend is busy; recent logs request timed out."
            time.sleep(0.5)
        except RequestException as e:
            last_error = str(e)
            break
    return {"error": last_error or "Unable to fetch recent logs."}


def fetch_upload_sessions(limit: int = 200):
    data = fetch_recent_logs(limit=limit)
    if "error" in data:
        return data
    uploads = data.get("uploads", [])
    ready_uploads = [u for u in uploads if u.get("status") == "ready"]
    return {"uploads": ready_uploads}


def fetch_chat_history(chat_id: str, limit: int = 200):
    try:
        resp = requests.get(
            f"{API_URL}/chat-history/{chat_id}",
            params={"limit": limit},
            headers=auth_headers(),
            timeout=45,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text}
    except RequestException as e:
        return {"error": str(e)}

st.sidebar.title("📁 Upload Data")
st.sidebar.title("🔐 Authentication")

with st.sidebar.form("signup_form"):
    signup_user = st.text_input("New Username")
    signup_pass = st.text_input("New Password", type="password")
    signup_clicked = st.form_submit_button("Create User")

if signup_clicked:
    try:
        create_resp = requests.post(
            f"{API_URL}/users",
            json={"username": signup_user, "password": signup_pass},
            timeout=15,
        )
        if create_resp.status_code in (200, 201):
            st.sidebar.success("User created. Now login to get bearer token.")
        else:
            st.sidebar.error(f"Create user failed: {create_resp.text}")
    except Exception as e:
        st.sidebar.error(f"Create user error: {e}")

with st.sidebar.form("login_form"):
    login_user = st.text_input("Username", value=st.session_state.auth_user)
    login_pass = st.text_input("Password", type="password")
    login_clicked = st.form_submit_button("Login")

if login_clicked:
    try:
        token_resp = requests.post(
            f"{API_URL}/token",
            data={"username": login_user, "password": login_pass},
            timeout=15,
        )
        if token_resp.status_code == 200:
            token_data = token_resp.json()
            st.session_state.access_token = token_data["access_token"]
            st.session_state.auth_user = login_user
            st.sidebar.success(f"Logged in as {login_user}")
        else:
            st.sidebar.error(f"Login failed: {token_resp.text}")
    except Exception as e:
        st.sidebar.error(f"Login error: {e}")

if st.session_state.access_token:
    st.sidebar.caption(f"Authenticated as: {st.session_state.auth_user}")
else:
    st.sidebar.warning("Login required to upload files and chat.")

st.sidebar.title("🧠 Resume Chat")
if st.session_state.access_token:
    session_data = fetch_upload_sessions(limit=200)
    if "error" in session_data:
        st.sidebar.caption(f"Could not load sessions: {session_data['error']}")
    else:
        uploads = session_data.get("uploads", [])
        if not uploads:
            st.sidebar.caption("No ready sessions available yet.")
        else:
            option_map = {}
            labels = []
            for upload in uploads:
                chat_id = upload.get("chat_id", "")
                filename = upload.get("filename", "unknown")
                created_at = (upload.get("created_at") or "")[:19]
                label = f"{filename} | {chat_id[:8]} | {created_at}"
                labels.append(label)
                option_map[label] = upload

            default_idx = 0
            if st.session_state.chat_id:
                for i, label in enumerate(labels):
                    if option_map[label].get("chat_id") == st.session_state.chat_id:
                        default_idx = i
                        break

            selected_label = st.sidebar.selectbox(
                "Previous sessions",
                options=labels,
                index=default_idx,
                key="resume_session_selector",
            )
            if st.sidebar.button("Load Selected Session"):
                selected_upload = option_map[selected_label]
                selected_chat_id = selected_upload.get("chat_id")
                history_data = fetch_chat_history(selected_chat_id, limit=250)
                if "error" in history_data:
                    st.sidebar.error(f"Failed to load chat history: {history_data['error']}")
                else:
                    st.session_state.chat_id = selected_chat_id
                    st.session_state.file_meta = {
                        "type": history_data.get("file_type", selected_upload.get("file_type")),
                        "name": history_data.get("filename", selected_upload.get("filename")),
                    }
                    st.session_state.messages = history_data.get("messages", [])
                    st.sidebar.success("Session restored.")

uploaded_file = st.sidebar.file_uploader(
    "Support: PDF, DOC/DOCX, PPT/PPTX, TXT, CSV, XLS/XLSX, DB/SQLite, JPG/PNG", 
    type=[
        "pdf", "doc", "docx", "ppt", "pptx", "txt", "csv",
        "xls", "xlsx", "db", "sqlite", "sqlite3",
        "png", "jpg", "jpeg"
    ]
)

st.sidebar.title("⚙️ Model Settings")
model_options = fetch_models()
if not model_options:
    model_options = [
        {
            "id": "gemini-2.5-flash",
            "label": "Gemini 2.5 Flash",
            "provider_label": "Google AI Studio",
            "use_for": "Best default for most chats (balanced quality + speed).",
            "details": "Fallback list used because backend /models is unavailable.",
        }
    ]

model_labels = [f"{m['label']} ({m['provider_label']})" for m in model_options]
for i, m in enumerate(model_options):
    if not m.get("enabled", True):
        model_labels[i] = f"{model_labels[i]} - Not Configured"

default_index = 0
for i, m in enumerate(model_options):
    if m.get("enabled", True):
        default_index = i
        break

selected_label = st.sidebar.selectbox("Model", options=model_labels, index=default_index)
selected_model = model_options[model_labels.index(selected_label)]

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.2,
    step=0.1,
)
st.sidebar.caption(f"Use case: {selected_model['use_for']}")
st.sidebar.caption(selected_model["details"])
st.sidebar.caption(f"Free tier: {selected_model.get('free_tier', 'varies by provider/account')}")
st.sidebar.caption(
    f"Category: {selected_model.get('category', 'text')} | "
    f"Vision: {selected_model.get('supports_vision', 'no')} | "
    f"OCR: {selected_model.get('ocr_mode', 'Local OCR')}"
)
if not selected_model.get("enabled", True):
    st.sidebar.warning("This provider key is missing in .env. Requests will fail until configured.")

st.sidebar.title("📊 My Logs")
log_limit = st.sidebar.selectbox("Recent rows", options=[10, 20, 30, 50], index=1)
show_logs = st.sidebar.checkbox("Show user-wise logs", value=True)
if show_logs and st.session_state.access_token:
    summary = fetch_logs_summary()
    if "error" in summary:
        st.sidebar.error(f"Logs summary failed: {summary['error']}")
    else:
        uploads = summary.get("uploads", {})
        chat = summary.get("chat", {})
        st.sidebar.caption(
            f"Uploads: {uploads.get('total', 0)} | Ready: {uploads.get('ready', 0)} | Failed: {uploads.get('failed', 0)}"
        )
        st.sidebar.caption(
            f"Turns: {chat.get('total_turns', 0)} | Avg latency: {chat.get('avg_latency_ms')} ms | Tokens(est): {chat.get('total_tokens_estimated', 0)}"
        )
        st.sidebar.caption(
            f"Eval -> Relevancy: {chat.get('avg_answer_relevancy')} | "
            f"Faithfulness: {chat.get('avg_faithfulness')} | "
            f"Context Precision: {chat.get('avg_contextual_precision')}"
        )

if uploaded_file and st.sidebar.button("Process File"):
    with st.spinner("Uploading file..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            resp = requests.post(
                f"{API_URL}/upload",
                files=files,
                headers=auth_headers(),
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data["file_type"] == "doc":
                    with st.spinner("Analyzing (OCR & Indexing) in background..."):
                        status_data = wait_for_ingestion(data["chat_id"], timeout_seconds=1200, poll_interval=5.0)
                    if status_data.get("state") != "ready":
                        st.sidebar.error(f"Indexing failed: {status_data.get('error', 'unknown error')}")
                    else:
                        st.session_state.chat_id = data["chat_id"]
                        st.session_state.file_meta = {
                            "type": data["file_type"],
                            "name": data["filename"]
                        }
                        st.session_state.messages = []
                        st.sidebar.success("Ready!")
                else:
                    st.session_state.chat_id = data["chat_id"]
                    st.session_state.file_meta = {
                        "type": data["file_type"],
                        "name": data["filename"]
                    }
                    st.session_state.messages = []
                    st.sidebar.success("Ready!")
            else:
                st.sidebar.error(f"Error: {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

st.title("🤖 Chat with your Data")

if st.session_state.access_token and show_logs:
    with st.expander("My Logs & Metrics (DB-backed)", expanded=False):
        recent = fetch_recent_logs(limit=log_limit)
        if "error" in recent:
            st.error(f"Failed to load logs: {recent['error']}")
        else:
            st.subheader("Recent Upload Sessions")
            st.dataframe(recent.get("uploads", []), width="stretch")
            st.subheader("Recent Chat Turns")
            st.dataframe(recent.get("turns", []), width="stretch")

if st.session_state.chat_id:
    # Display History
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_resp = ""
            
            params = {
                "chat_id": st.session_state.chat_id,
                "query": prompt,
                "file_type": st.session_state.file_meta["type"],
                "filename": st.session_state.file_meta["name"],
                "model": selected_model["id"],
                "temperature": temperature,
            }
            
            with requests.get(
                f"{API_URL}/chat",
                params=params,
                headers=auth_headers(),
                stream=True,
                timeout=(15, None),
            ) as r:
                if r.status_code != 200:
                    full_resp = f"Error: {r.text}"
                else:
                    try:
                        for chunk in r.iter_content(chunk_size=None):
                            if chunk:
                                text = chunk.decode("utf-8", errors="ignore")
                                full_resp += text
                                placeholder.write(full_resp + "▌")
                    except ChunkedEncodingError:
                        if not full_resp.strip():
                            full_resp = "Error: Response stream ended prematurely."
                if not full_resp.strip():
                    full_resp = "Error: Model returned an empty response. Please retry."
            placeholder.write(full_resp)
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
else:
    st.info("👈 Upload a document or database to start.")
