# Change Log

## Fixed Bugs

1. Added missing export cleanup helpers.
   - Implemented `cleanup_exports()` so `/generate-file` and chart rendering no longer fail with `NameError`.
   - Added `safe_export_stem()` to keep export filenames filesystem-safe.
   - References: `app.py:532-561`, `app.py:1054-1105`, `app.py:1161-1193`, `app.py:1255`.

2. Closed the conversation data leak in `/generate-file`.
   - Added an ownership check before loading message history for `req.convo_id`.
   - Now the endpoint returns `404` instead of exposing another user's conversation context.
   - References: `app.py:1225-1239`.

3. Blocked filename path traversal in upload flows.
   - Added `sanitize_filename()` and applied it to both `/upload` and `/fetch-url`.
   - Filenames with path components such as `../../file.txt` are now rejected.
   - References: `app.py:548-556`, `app.py:937-998`.

4. Made `mode="general"` truly bypass vector retrieval.
   - Moved mode parsing ahead of retrieval and skipped embeddings/Qdrant lookup when the request is general-only.
   - This removes an unnecessary dependency on the vector stack for LLM-only prompts.
   - References: `app.py:772-795`, `app.py:850-854`.

5. Matched export backend behavior to the frontend menu.
   - Expanded `/export/{conv_id}` to support `md`, `json`, `txt`, `csv`, and `docx`.
   - Added explicit validation for unsupported formats instead of silently falling back.
   - References: `app.py:1032-1105`.

6. Fixed the chat response contract for performance metrics.
   - Extended `ChatResponse` with a `performance` field and returned `result["performance"]` from `/chat`.
   - The existing frontend performance display can now receive real data.
   - References: `app.py:520-524`, `app.py:895-900`.

## Verification

- Ran `python3 -m py_compile app.py` successfully.

## Web Search Upgrade

1. Replaced DuckDuckGo/DDGS web search with Brave Search API.
   - Added Brave configuration env vars and switched the backend search call to `https://api.search.brave.com/res/v1/web/search`.
   - Added support for a per-request web result count from the chat UI.

2. Added Crawl4AI extraction and sanitization for web results.
   - Search results are now crawled, converted to filtered markdown, and sanitized before being passed into the RAG prompt.
   - Added a lightweight prompt-injection scrubber that removes suspicious instruction-like lines from fetched web content.

3. Returned web citations to the user.
   - Web results are included in the chat response source list as clickable URLs so users can verify them manually.

4. Updated Docker/runtime dependencies.
   - Removed DDGS dependencies, added `crawl4ai==0.8.0`, and updated the Docker build to install Playwright Chromium for Crawl4AI.
