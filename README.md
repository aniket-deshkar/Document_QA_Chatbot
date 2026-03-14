# Document QA Knowledge Base Chatbot

A production-style Retrieval-Augmented Generation (RAG) system permitting users to upload documents and perform conversational Q&A against the extracted contents. 

## System Capabilities

![UI Preview](output%20screenshots/Authentication%20Output.png)
![Chat Interface](output%20screenshots/Resume%20previous%20chat%20and%20csv.png)

- **Stateful Conversational UI**: Features a dedicated `frontend` module to interactively chat with your specific document knowledge bases.
- **Robust Backend Infrastructure**: Supports uploading files to `uploads/`, chunking, and embedding.
- **Vector Search (ChromaDB)**: Persists document embeddings locally in `chroma_db/` for efficient vector similarity searches.
- **System Telemetry & Caching**: Employs `system.db` for metadata management, comprehensive `logs/`, and a local `.model_cache` to reduce redundant embedding API calls and accelerate response times.

## Setup Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide the API keys for the embedded model in a `.env` file.

Start the backend and frontend components as specified in their respective entry scripts to begin knowledge extraction.

*Designed as a scalable blueprint for enterprise document analysis and corporate knowledge augmentation.*
