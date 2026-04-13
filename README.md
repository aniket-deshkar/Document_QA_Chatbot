# Document QA Knowledge Base Chatbot

Production-style Retrieval-Augmented Generation (RAG) application for uploading enterprise documents, indexing them into a vector store, and answering questions through a conversational UI.

This project is designed to demonstrate backend API design, document ingestion pipelines, vector retrieval, conversational memory, and a simple product-facing frontend in one repo.

## Why this project

Teams often store useful knowledge in PDFs, spreadsheets, images, and internal documents that are hard to search quickly. This application turns those files into a searchable knowledge base and exposes the result through a chat workflow.

## Key capabilities

![Authentication](output%20screenshots/Authentication%20Output.png)
![Chat Interface](output%20screenshots/Resume%20previous%20chat%20and%20csv.png)

- Upload and process multiple document types
- Chunk and embed content for semantic retrieval
- Store embeddings locally with ChromaDB
- Chat over indexed documents with context-aware responses
- Maintain application metadata and logs for debugging
- Provide a lightweight frontend for end-user interaction

## Tech stack

- Python
- Streamlit frontend
- ChromaDB
- Environment-based configuration
- Local persistence for metadata, caches, and logs

## High-level architecture

1. User uploads one or more files through the frontend.
2. The backend loads and parses the documents.
3. Content is chunked and converted to embeddings.
4. Chunks are stored in ChromaDB for similarity search.
5. A retrieval pipeline selects relevant context for each question.
6. The response is generated and returned through the chat UI.

## Repository structure

```text
frontend/                Streamlit application
backend/main.py          Backend entrypoint
backend/rag_engine.py    Retrieval and response orchestration
backend/rag_loaders.py   Document loading and parsing
backend/rag_embedding.py Embedding pipeline
backend/rag_rerank.py    Retrieval refinement
backend/sql_engine.py    Structured storage helpers
output screenshots/      UI and workflow screenshots
```

## What this demonstrates

- Building a multi-step RAG pipeline instead of a single prompt wrapper
- Separating frontend and backend responsibilities
- Managing local vector persistence and supporting assets
- Structuring an AI application with reusable modules rather than notebook-only code
- Thinking about observability through logs, caches, and metadata storage

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with the model credentials and configuration required by your embedding and generation providers.

Then start the application entrypoints you want to use:

- backend service from `backend/main.py`
- frontend UI from `frontend/app.py`

## Portfolio value

This project is a good showcase for companies looking for engineers who can combine:

- applied LLM product work
- backend service design
- retrieval systems
- practical developer experience and UI thinking

## Suggested next improvements

- Add automated tests for loaders and retrieval logic
- Add Docker Compose for one-command local startup
- Add evaluation metrics for answer quality and retrieval relevance
- Add CI checks for formatting and dependency validation
