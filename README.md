# Protocol Atlas

Protocol Atlas is a RAG application for turning protocol documents into searchable, cited working knowledge.

## Current slice

- Next.js TypeScript UI with PDF selection, protocol library, source excerpts, and grounded mock Q&A.
- FastAPI service with PDF upload metadata extraction, a provider-neutral retrieval core, and `/query` responses with page citations.
- Deterministic mock mode so the project runs without model or vector-store credentials.
- Synthetic development records in `backend/data/protocols.json`; they are clearly marked as unvalidated sample data.
- SQLite persistence for uploaded protocol metadata and source chunks.
- OpenAI-compatible answer and embedding provider boundaries, enabled with environment variables and safe mock fallback.
- Hybrid retrieval contracts for BGE-M3 dense, sparse lexical, and ColBERT multi-vectors in Qdrant.
- Provenance-aware general scientific context, returned separately from protocol citations.
- Optional `API_KEY` protection for upload and query endpoints.

## Run locally

Frontend:

```bash
npm run dev
```

Backend, in another terminal:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The frontend is available at `http://localhost:3000` and the API at `http://localhost:8000`.

## Configure providers

Copy `.env.example` to `.env`. Leave provider keys empty for local mock mode. For a real OpenAI-compatible service, set `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_API_KEY`, and `RETRIEVAL_MODE=embeddings`. The frontend can send `NEXT_PUBLIC_API_KEY` when `API_KEY` is enabled. Without those values, keyword retrieval and deterministic mock answers are used.

### Hugging Face

Hugging Face uses the OpenAI-compatible router for chat and its feature-extraction route for embeddings. Set `HF_API_TOKEN` and uncomment the Hugging Face block in `.env.example`. Use a model supported by your Hugging Face account and selected Inference Provider.

## Retrieval modes

`RETRIEVAL_MODE=keyword` is the lightweight local path. The full architecture uses `FlagEmbedding` to generate BGE-M3 dense, sparse, and ColBERT vectors, then stores and fuses them in Qdrant with reciprocal-rank fusion. Enable it with `HYBRID_ENCODER=bge-m3`, `QDRANT_URL`, and the production retrieval dependencies. The adapter code is in `backend/app/bge_encoder.py` and `backend/app/hybrid.py`.

## Container deployment

```bash
cp .env.example .env
docker compose up --build
```

The compose volume keeps the SQLite catalog across container restarts. In hosted production, put the API behind HTTPS, use a managed database/vector store, and keep secrets in the platform secret manager rather than committing `.env`.

## Test

```bash
cd backend
pytest
```

The next implementation step is replacing the local keyword index with a managed vector store and adding user/session authentication for multi-user deployments.

General context provenance is available at `/knowledge/sources`. The current entries are curated local notes and are intentionally marked `Review required before external publication`; replace them with reviewed, licensed public sources before presenting them as open-source evidence.

### Public knowledge connector

The backend includes an optional Europe PMC open-access connector. To enable it locally, set:

```env
PUBLIC_KNOWLEDGE_PROVIDER=europepmc
```

The connector searches open-access records for general context only. It fails open when the service is unavailable, does not replace protocol excerpts, and returns the record identifier, URL, source, and license metadata under `knowledge_sources`. Review article-level licensing before redistributing retrieved text in a hosted product.
