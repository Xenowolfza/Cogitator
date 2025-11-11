## Warhammer Rules Assistant — AI agent guidance

This file gives concise, repo-specific guidance for AI coding agents working on this project.

- Project type: Streamlit single-page app (entry `app.py`) that builds an in-memory FAISS index from Warhammer PDF rules and answers questions using OpenAI chat+embeddings.
- Key supporting modules:
  - `rules_updater.py` — scrapes Warhammer download pages and downloads PDFs.
  - `rules_engine.py` — alternative loader using LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter` (used for experiments / batch pipelines).

Important patterns and conventions
- Secrets: Use Streamlit secrets or environment variables. Keys (used in `app.py` / README): `OPENAI_API_KEY`, `APP_ACCESS_PIN`, `DEFAULT_MODEL`, `EMBEDDING_MODEL`, `VISION_MODEL`, `SYSTEM_PROMPT`, `WELCOME_MESSAGE`.
- Embeddings: `EMBEDDING_MODEL` defaults to `text-embedding-ada-002`. `EMBEDDING_DIM` is inferred in `app.py` (1536 or 3072 depending on name); preserve this heuristic.
- FAISS usage: Indexes are created per ruleset via `get_faiss_store()` and `create_faiss_resource(system)`. Vectors are normalized before adding — preserve normalization in `add_to_faiss`.
- Text extraction: `app.py` uses `PyPDF2` text extraction with a simple OCR fallback via OpenAI Vision. `rules_engine.py` demonstrates a LangChain-based approach — don't conflate them without testing.
- Caching: Streamlit caching decorators are used: `@st.cache_resource` (for clients/index store) and `@st.cache_data` (for built indices). Respect these when refactoring to avoid surprising state resets.
- Error handling: The code checks OpenAI `APIError` codes (e.g., `'insufficient_quota'`, 429). Keep this exact behavior for user-visible messages.

How to run locally (examples from README)
1. Create venv and install requirements:
   python -m venv venv
   # activate accordingly on Windows: `venv\Scripts\Activate.ps1`
   pip install -r requirements.txt
2. Add secrets in `.streamlit/secrets.toml` or environment.
3. Run:
   streamlit run app.py

Files to inspect for context and examples
- `app.py` — primary UI, PDF download list (`WARHAMMER_PDFS`), embedding / FAISS logic, session_state keys used (`last_indexed`, `current_system`, `image_text`). Use it as the canonical example for UI behavior.
- `rules_updater.py` — shows how download links are scraped and how PDFs are downloaded to disk. Useful for creating longer-running update jobs.
- `rules_engine.py` — shows LangChain-based splitting used in experiments.
- `requirements.txt` — lists runtime dependencies (Streamlit, openai, faiss-cpu, PyPDF2, beautifulsoup4).

Examples of repo-specific tasks for agents
- To change fetch sources: update `WARHAMMER_PDFS` in `app.py`. Do not change URLs without validating the remote file; prefer adding new entries and testing download.
- To add persistent index storage: follow existing `get_faiss_store()` per-system pattern, but migrate storage behind the same API (`create_faiss_resource`, `add_to_faiss`, `search_faiss`).
- To improve chunking: test changes against `chunk_text` in `app.py` (simple word-based) and compare behavior with LangChain splitter in `rules_engine.py`.

Quick notes for safe edits
- Preserve session_state keys and cache decorators or behavior will change unexpectedly in Streamlit UX.
- Keep OpenAI error handling and retry/backoff behavior; users depend on clear quota/rate-limit messages.
- When modifying embedding dims or model names, update both the `EMBEDDING_MODEL` default and the `EMBEDDING_DIM` inference logic.

If anything here is unclear or you'd like more examples (tests, a sample small PDF, or a CI job to validate installs), tell me which section to expand.
