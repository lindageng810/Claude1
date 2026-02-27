# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

The server must be started from the `backend/` directory. On this machine, two extra env vars are required due to network constraints:

```bash
cd backend
unset http_proxy && unset https_proxy && HF_ENDPOINT=https://hf-mirror.com uv run uvicorn app:app --port 8000
```

- `unset http_proxy/https_proxy` — clears a malformed system proxy (`http:127.0.0.1:15236`) that breaks connections
- `HF_ENDPOINT=https://hf-mirror.com` — required to load the `all-MiniLM-L6-v2` embedding model (HuggingFace is blocked; use mirror)
- Omit `--reload` in production to avoid subprocess output capture issues with the reloader

Shortcut alias (configured in `~/.zshrc`): `run-claude`

Access at `http://localhost:8000` — API docs at `http://localhost:8000/docs`.

## Environment

Create `backend/.env`:
```
DEEPSEEK_API_KEY=your_key_here
```

Install dependencies (run from project root):
```bash
unset http_proxy && unset https_proxy && uv sync
```

Git push requires the proxy:
```bash
git push origin main   # global proxy already configured via git config --global http.proxy
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system with a full-stack layout:

```
backend/    — FastAPI server + all RAG logic
frontend/   — Vanilla HTML/CSS/JS (served as static files by FastAPI)
docs/       — Course .txt files loaded into ChromaDB on startup
```

### Request Flow

1. **Frontend** (`script.js`) — `sendMessage()` POSTs `{query, session_id}` to `/api/query`
2. **FastAPI** (`app.py`) — validates request, creates/reuses session, delegates to `RAGSystem`
3. **RAGSystem** (`rag_system.py`) — wraps query, fetches conversation history, calls `AIGenerator`
4. **AIGenerator** (`ai_generator.py`) — calls DeepSeek API with tool definitions; if `finish_reason == "tool_calls"`, executes search tool and makes a second API call with results
5. **CourseSearchTool** (`search_tools.py`) — calls `VectorStore.search()` which queries ChromaDB and returns top-5 chunks
6. **Response** — sources + answer returned to frontend; session history updated

### AI / Tool Calling

- Uses `openai` SDK pointed at DeepSeek: `OpenAI(api_key=..., base_url="https://api.deepseek.com")`
- Model: `deepseek-chat`, temperature 0, max_tokens 800
- Tool definitions use **OpenAI function-calling format** (`{"type": "function", "function": {...}}`), not Anthropic format
- `ToolManager.register_tool()` reads the tool name from `tool_def["function"]["name"]`

### Vector Store (ChromaDB)

Two collections in `./chroma_db`:
- `course_catalog` — course metadata, used for fuzzy course-name matching
- `course_content` — text chunks with metadata (`course_title`, `lesson_number`, `chunk_index`)

Embedding model: `all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction`.

### Document Format

Course `.txt` files must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content>

Lesson 1: <title>
...
```

Chunks are ~800 chars with 100-char sentence-level overlap. The first chunk of each lesson is prefixed with `"Lesson N content: ..."` for context; the last lesson's chunks are prefixed with `"Course <title> Lesson N content: ..."`.

### Session Management

Sessions are stored **in memory only** (not persisted). `SessionManager` keeps the last 10 messages (5 turns) per session. Sessions are identified by `"session_N"` strings managed by the frontend via `currentSessionId`.

## Key Config (`backend/config.py`)

| Key | Default | Notes |
|-----|---------|-------|
| `DEEPSEEK_MODEL` | `deepseek-chat` | Change here to swap models |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Changing requires clearing `chroma_db/` |
| `CHUNK_SIZE` | 800 | Changing requires re-ingesting all docs |
| `CHUNK_OVERLAP` | 100 | |
| `MAX_RESULTS` | 5 | Top-K chunks returned per search |
| `MAX_HISTORY` | 2 | Conversation pairs passed to AI |
| `CHROMA_PATH` | `./chroma_db` | Relative to `backend/` |
