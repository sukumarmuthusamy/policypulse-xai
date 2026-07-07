# PolicyPulse

**Enterprise Policy Copilot** — a lean, production-oriented RAG application that answers natural-language questions about company policy PDFs with hybrid retrieval, explainable AI auditing, and live operational telemetry.

PolicyPulse is built as a **dual-service architecture** (FastAPI backend + Streamlit frontend) designed for local development, Docker Compose deployment, and future Cloud Run scaling. It uses a **native Python tool-calling agent**—no LangGraph, LangChain orchestration, or local embedding runtimes—keeping containers lightweight and cold starts fast.

---

## Features

| Capability | Description |
|------------|-------------|
| **Hybrid Vector + BM25 Sparse Search (RRF)** | Combines FAISS dense retrieval with BM25 keyword search, fused via Reciprocal Rank Fusion. Eliminates the dense-retrieval blind spot on short, keyword-heavy queries (e.g., hourly pay rates, specific dollar amounts). |
| **Explainable AI (XAI) Inspector** | Per-response audit panel showing trace ID, latency, tool calls, raw intent, and retrieved chunks with normalized match scores and source citations. |
| **Live Telemetry Sidebar** | Real-time dashboard of model provider, index health, chunk/vector counts, and rolling p50 / last-query latency from structured execution traces. |
| **Password-Protected Demo Gateway** | Streamlit access is gated by an environment-driven `DEMO_PASSWORD` to prevent unauthorized LLM token consumption in public demo deployments. |
| **Dynamic PDF Upload** | Upload policy documents via the UI; indexes rebuild automatically with hot-reload—no container restart required. |
| **Upload Size Guardrail (10MB)** | Backend `/upload` enforces a strict 10MB limit and returns HTTP 413 for oversized PDFs to keep deployments lightweight and cost-controlled. |
| **Provider-Agnostic LLM Layer** | Switch between **Gemini** and **OpenAI** for chat and embeddings via environment configuration. |
| **Structured JSONL Tracing** | Every agent invocation appends a machine-readable trace for observability, debugging, and compliance auditing. |
| **Lean Container Footprint** | API-based embeddings only—no `sentence-transformers` or local model runtimes. Optimized for serverless and free-tier cloud deployment. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend  :8501                           │
│   Chat UI  ·  Telemetry Sidebar  ·  XAI Inspector  ·  PDF Upload        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP
┌───────────────────────────────▼─────────────────────────────────────────┐
│                      FastAPI Backend  :8000                              │
│   /health  ·  /metadata  ·  /agent  ·  /upload                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Agent Orchestrator  →  LLM Adapters (Gemini / OpenAI)                   │
│                     →  retrieve_policy_context (Hybrid RAG Tool)         │
│                     →  JSONL Tracer                                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  data/policies/          storage/              External APIs
  (Source PDFs)     faiss.index · bm25.pkl    Gemini / OpenAI
                    chunks.json · traces
```

**Retrieval pipeline (per query):**

1. **Dense (FAISS)** — top 4 chunks by cosine similarity (L2-normalized vectors, inner product index)
2. **Sparse (BM25)** — top 4 chunks by keyword relevance (numerals preserved in tokenization)
3. **RRF merge** — `score += 1 / (60 + rank)` for each appearance in either list
4. **Deduplicate & rank** — return top *k* fused chunks (default 5), scores normalized to 0–1 for the UI

---

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| API Server | FastAPI, uvicorn |
| UI | Streamlit, httpx |
| Dense Retrieval | faiss-cpu, numpy |
| Sparse Retrieval | rank-bm25 (BM25Okapi) |
| PDF Parsing | pypdf |
| LLM & Embeddings | google-generativeai (Gemini), openai |
| Configuration | pydantic-settings, python-dotenv |
| Testing | pytest (30 tests) |
| Containers | Docker, Docker Compose |

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (v2+)
- An API key for your chosen provider:
  - **Gemini** (default): [Google AI Studio](https://aistudio.google.com/apikey)
  - **OpenAI** (optional): [OpenAI Platform](https://platform.openai.com/api-keys)
- One or more policy PDF files to index

---

## Quick Start with Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/policypulse.git
cd policypulse
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your API credentials:

```bash
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_api_key_here
```

> **OpenAI users:** Set `MODEL_PROVIDER=openai`, `MODEL_NAME=gpt-4o-mini`, and `OPENAI_API_KEY`.
>
> **Demo protection:** Set `DEMO_PASSWORD` in `.env` to secure public demos. If omitted, the frontend uses `suku-pulse` as the fallback password.

### 3. Add policy documents

Place your PDF files in the policies directory:

```bash
cp /path/to/your/policies/*.pdf data/policies/
```

### 4. Build the search index

The backend requires a pre-built FAISS + BM25 index before it can serve queries. Build it once using the backend container:

```bash
docker compose run --rm backend python scripts/build_index.py
```

This scans `data/policies/`, chunks each PDF (800 characters, 150 overlap), embeds via your configured provider API, and writes artifacts to `storage/`:

| Artifact | Purpose |
|----------|---------|
| `storage/faiss.index` | Dense vector index |
| `storage/bm25.pkl` | Sparse BM25 index |
| `storage/chunks.json` | Chunk metadata sidecar |

> **Note:** Both indexes must be rebuilt after adding hybrid search support if only a legacy FAISS index exists.

### 5. Start the application

```bash
docker compose up --build
```

### 6. Access the services

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | [http://localhost:8501](http://localhost:8501) | Chat, telemetry sidebar, XAI Inspector, PDF upload |
| **API** | [http://localhost:8000](http://localhost:8000) | REST endpoints |
| **Health check** | [http://localhost:8000/health](http://localhost:8000/health) | Liveness probe |
| **Metadata** | [http://localhost:8000/metadata](http://localhost:8000/metadata) | Provider, index stats, latency rollups |

### 7. Stop the application

```bash
docker compose down
```

Persistent data (`data/policies/`, `storage/`) survives container restarts via volume mounts.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Returns `{"status": "ok"}` |
| `/metadata` | `GET` | Provider info, index status, chunk/vector counts, rolling latency |
| `/agent` | `POST` | `{"query": "..."}` → answer with full execution trace |
| `/upload` | `POST` | Multipart PDF upload (max 10MB) → rebuild index → hot-reload; oversized files return HTTP 413 |

**Example agent request:**

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?"}'
```

---

## Retrieval Optimization & Performance Tuning

PolicyPulse exposes a single RAG tool to the agent: `retrieve_policy_context(query, top_k?)`. The **`top_k` parameter is the primary lever for balancing retrieval recall against LLM token consumption**.

### How `top_k` works

`top_k` controls the **maximum number of policy passages** returned after hybrid fusion (dense FAISS + sparse BM25 merged via RRF). The default is **5** (`HYBRID_FINAL_K`).

Internally, hybrid search always retrieves **4 dense** and **4 sparse** candidates, then fuses and deduplicates them. The final `top_k` caps how many of those fused results are passed to the LLM as context.

### The recall problem (and the fix)

Short, keyword-heavy queries expose a known weakness of **dense-only retrieval**. For example, a query like *"what's the pay per hour?"* can cause FAISS to surface semantically related but **incorrect** passages—generic reimbursement language—while **excluding** the chunk containing the exact **"$80–$105"** salary figure.

**Root cause:** Dense embeddings prioritize broad semantic similarity over exact token overlap. Short queries carry weak semantic signal; numerals and specific phrases are underrepresented in embedding space.

**Solution:** Hybrid search + RRF ensures BM25 keyword matches compete on equal footing with dense results. Increasing `top_k` further widens the fused context window so critical keyword-aligned chunks are not crowded out.

### Tuning guidance

| Scenario | Recommended `top_k` | Trade-off |
|----------|---------------------|-----------|
| General policy questions (paragraph-length) | **4–5** (default) | Balanced recall and token cost |
| Short, keyword-heavy queries (rates, dates, IDs) | **5–8** | Higher recall; slightly more LLM input tokens |
| Maximum coverage for complex multi-topic questions | **8–10** | Near-complete elimination of false negatives; modest token increase |

### Token cost impact

Each additional retrieved chunk adds roughly **150–250 tokens** to the tool-response context (800-character chunks with overlap). Raising `top_k` from 4 to 8 typically adds **~600–1,000 input tokens** per retrieval call—a **minor increase** relative to full conversation history, but meaningful at scale.

The LLM selects `top_k` dynamically via tool calling. For consistently short or numeric queries, the model tends to request higher values after observing missed answers at lower settings.

### Operational checklist

1. **Rebuild the index** after adding or updating PDFs (`python scripts/build_index.py` or UI upload).
2. **Verify hybrid indexes exist** — both `faiss.index` and `bm25.pkl` must be present in `storage/`.
3. **Inspect the XAI Inspector** after queries to confirm retrieved chunks and match scores.
4. **Monitor telemetry** — rising p50 latency may indicate embedding API slowness, not retrieval tuning issues.

---

## Local Development (without Docker)

```bash
# Virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env               # configure API keys

# Build index
python scripts/build_index.py

# Terminal 1 — Backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
BACKEND_URL=http://127.0.0.1:8000 streamlit run frontend/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v
```

### Verification scripts

```bash
python scripts/verify_index.py     # Index load + search smoke test
python scripts/verify_agent.py     # End-to-end agent smoke test
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `gemini` | LLM provider: `gemini` or `openai` |
| `MODEL_NAME` | `gemini-2.5-flash` | Chat model override |
| `GEMINI_API_KEY` | — | Required when `MODEL_PROVIDER=gemini` |
| `OPENAI_API_KEY` | — | Required when `MODEL_PROVIDER=openai` |
| `DEPLOYMENT_TARGET` | `local` | Deployment context: `local`, `docker`, `cloud-run` |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Frontend → backend URL |
| `DEMO_PASSWORD` | `suku-pulse` (fallback) | Streamlit demo gateway password; override in `.env` for production/demo deployments |
| `POLICIES_DIR` | `data/policies` | Source PDF directory |
| `FAISS_INDEX_PATH` | `storage/faiss.index` | Dense index path |
| `BM25_INDEX_PATH` | `storage/bm25.pkl` | Sparse index path |
| `CHUNKS_PATH` | `storage/chunks.json` | Chunk metadata path |
| `LOG_PATH` | `structured_logs.jsonl` | JSONL trace log path |
| `MAX_TOOL_ITERATIONS` | `5` | Agent tool-calling loop cap |

---

## Project Structure

```
policypulse/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Environment-driven settings
│   ├── agents/                 # Orchestrator, tools, schemas
│   ├── services/               # LLM and embedding factories
│   ├── rag/hybrid.py           # RRF fusion and tokenization
│   └── observability/tracer.py # Structured JSONL tracing
├── frontend/
│   ├── streamlit_app.py        # Chat UI, telemetry, XAI Inspector
│   └── .streamlit/config.toml  # Streamlit framework configs
├── scripts/                    # Index build and verification CLIs
├── data/policies/              # Source PDFs (volume-mounted)
├── storage/                    # Persisted indexes and traces
├── tests/                      # pytest suite
├── docker-compose.yml
├── backend.Dockerfile
└── frontend.Dockerfile
```

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.
