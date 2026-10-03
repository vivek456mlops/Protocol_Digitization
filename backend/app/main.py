from pathlib import Path
import io
import json
import os
import re
from typing import Optional
from urllib.parse import unquote
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from dotenv import load_dotenv

from .rag import Chunk, EmbeddingRetriever, InMemoryRetriever
from .providers import configured_answer_provider, configured_embedding_provider
from .public_knowledge import configured_public_provider
from .storage import ProtocolStore

load_dotenv(Path(__file__).parents[2] / ".env")

app = FastAPI(title="Protocol Atlas API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parents[1] / "data" / "protocols.json"
with DATA_PATH.open(encoding="utf-8") as data_file:
    mock_data = json.load(data_file)
DOMAIN_PATH = Path(__file__).parents[1] / "data" / "domain_knowledge.json"
with DOMAIN_PATH.open(encoding="utf-8") as domain_file:
    domain_data = json.load(domain_file)

store = ProtocolStore(Path(os.getenv("PROTOCOL_DB_PATH", str(DATA_PATH.with_suffix(".db")))))
store.seed(mock_data["protocols"])
retriever = InMemoryRetriever(store.chunks())
answer_provider = configured_answer_provider()
embedding_provider = configured_embedding_provider()
public_knowledge_provider = configured_public_provider()
retrieval_mode = os.getenv("RETRIEVAL_MODE", "keyword")
semantic_retriever = None


def get_semantic_retriever() -> Optional[EmbeddingRetriever]:
    global semantic_retriever, retrieval_mode
    if retrieval_mode != "embeddings":
        return None
    if semantic_retriever is None:
        try:
            semantic_retriever = EmbeddingRetriever(store.chunks(), embedding_provider)
        except Exception:
            retrieval_mode = "keyword-fallback"
            return None
    return semantic_retriever


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    protocol: Optional[str] = None


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    configured_key = os.getenv("API_KEY")
    if configured_key and x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="A valid API key is required.")


def refresh_retriever() -> None:
    retriever.chunks = store.chunks()
    if semantic_retriever:
        semantic_retriever.replace(retriever.chunks)


def protocol_metadata() -> list[dict]:
    return store.list_protocols()


def domain_context_entries(question: str) -> list[dict]:
    normalized = question.lower()
    matches = [
        entry
        for entry in domain_data["entries"]
        if any(topic in normalized for topic in entry["topics"])
    ]
    if "wash" in normalized and "column" in normalized:
        wash_note = domain_data["entries"][0]
        if wash_note not in matches:
            matches.append(wash_note)
    if any(term in normalized for term in ("increase", "decrease", "change", "more", "less", "optimization")):
        deviation_note = domain_data["entries"][1]
        if deviation_note not in matches:
            matches.append(deviation_note)
    return matches


def domain_context(question: str) -> list[str]:
    return [entry["text"] for entry in domain_context_entries(question)]


def domain_provenance(question: str) -> list[dict]:
    return [{"id": entry["id"], **entry["provenance"]} for entry in domain_context_entries(question)]


def public_context(question: str) -> list[dict]:
    if public_knowledge_provider is None:
        return []
    try:
        return public_knowledge_provider.search(question)
    except Exception:
        return []


def is_conversational(question: str) -> bool:
    normalized = " ".join(question.lower().strip().split()).rstrip("!?.")
    return bool(re.match(r"^h+i+(?:[\s,].*)?$", normalized)) or bool(re.match(r"^he+y+(?:[\s,].*)?$", normalized)) or normalized in {"hello", "good morning", "good afternoon", "good evening", "thanks", "thank you", "who are you"}


def is_protocol_catalog_request(question: str) -> bool:
    normalized = question.lower()
    protocol_word = "protocol" in normalized or "prtocol" in normalized or "protcol" in normalized
    list_word = any(term in normalized for term in ("list", "lis ", "available", "what do you have", "names"))
    return protocol_word and list_word


def unit_conversion_answer(question: str, sources: list[Chunk]) -> Optional[str]:
    """Answer simple unit conversions only when the source contains the amount."""
    if "liter" not in question.lower() or not sources:
        return None
    for source in sources:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(micro|µ|u)l", source.text.lower())
        if match:
            microliters = float(match.group(1))
            liters = microliters / 1_000_000
            display = f"{liters:g} liters"
            return f"The protocol specifies {match.group(1)} microliters for the sample tube, which equals {display}. Source: page {source.page}."
    return None


def requested_protocol_detail(question: str) -> Optional[dict]:
    normalized = question.lower()
    if not any(term in normalized for term in ("detail", "details", "about", "describe", "information")):
        return None
    for protocol in protocol_metadata():
        if protocol["name"].lower() in normalized:
            return protocol
    return None


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Protocol Atlas API", "status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "0.3.0",
        "mode": "provider" if answer_provider.__class__.__name__ != "MockAnswerProvider" else "mock",
        "embedding_provider": embedding_provider.__class__.__name__,
        "retrieval_mode": retrieval_mode,
        "storage": "sqlite",
    }


@app.get("/protocols")
def protocols() -> dict:
    return {"synthetic": mock_data["synthetic"], "protocols": protocol_metadata()}


@app.get("/knowledge/sources")
def knowledge_sources() -> dict:
    return {"sources": [{"id": entry["id"], **entry["provenance"]} for entry in domain_data["entries"]]}


@app.get("/protocols/{protocol_name}/pages/{page}")
def source_page(protocol_name: str, page: int) -> dict:
    source = store.source(unquote(protocol_name), page)
    if source is None:
        raise HTTPException(status_code=404, detail="Source page not found.")
    return source


@app.post("/query")
def query(payload: Question, _: None = Depends(require_api_key)) -> dict:
    request_id = str(uuid4())
    if is_conversational(payload.question):
        return {
            "request_id": request_id,
            "answer": answer_provider.answer(payload.question, []),
            "message": "Conversational response; no protocol sources were used.",
            "sources": [],
            "mode": "conversation",
        }
    if is_protocol_catalog_request(payload.question):
        names = ", ".join(protocol["name"] for protocol in protocol_metadata())
        return {
            "request_id": request_id,
            "answer": f"Available protocols: {names}.",
            "message": "Catalog response; no retrieval sources were needed.",
            "sources": [],
            "mode": "catalog",
        }
    detail_protocol = requested_protocol_detail(payload.question)
    if detail_protocol:
        detail_chunks = [chunk for chunk in retriever.chunks if chunk.protocol == detail_protocol["name"]]
        detail_sources = detail_chunks[:4]
        return {
            "request_id": request_id,
            "answer": f"{detail_protocol['name']} is a {detail_protocol.get('category', 'protocol')} document with {detail_protocol['pages']} pages, version {detail_protocol.get('version', 'unknown')}, updated {detail_protocol.get('updated', 'recently')}. It contains {len(detail_chunks)} indexed source sections.",
            "sources": [{"protocol": chunk.protocol, "page": chunk.page, "text": chunk.text} for chunk in detail_sources],
            "mode": "catalog",
        }
    active_retriever = get_semantic_retriever() or retriever
    contextual_notes = domain_context(payload.question)
    contextual_provenance = domain_provenance(payload.question)
    public_entries = public_context(payload.question)
    all_context = contextual_notes + [entry["text"] for entry in public_entries]
    all_provenance = contextual_provenance + [
        {key: entry[key] for key in ("id", "source", "kind", "license", "url")}
        for entry in public_entries
    ]
    if payload.protocol:
        scoped = [chunk for chunk in active_retriever.chunks if chunk.protocol == payload.protocol]
        sources = (EmbeddingRetriever(scoped, embedding_provider).search(payload.question) if semantic_retriever else InMemoryRetriever(scoped).search(payload.question))
    else:
        sources = active_retriever.search(payload.question)
    if not sources:
        lexical_chunks = [chunk for chunk in retriever.chunks if not payload.protocol or chunk.protocol == payload.protocol]
        sources = InMemoryRetriever(lexical_chunks).search(payload.question)
    if not sources:
        if all_context:
            return {
                "request_id": request_id,
            "answer": answer_provider.answer(payload.question, [], all_context),
                "message": "General domain guidance; no matching protocol excerpt was found.",
                "sources": [],
            "knowledge_sources": all_provenance,
                "mode": "domain",
            }
        return {"request_id": request_id, "answer": None, "message": "No supporting evidence was found in the selected protocol.", "sources": [], "mode": "mock"}
    converted_answer = unit_conversion_answer(payload.question, sources)
    if converted_answer:
        return {
            "request_id": request_id,
            "answer": converted_answer,
            "sources": [{"protocol": source.protocol, "page": source.page, "text": source.text} for source in sources],
            "mode": "grounded",
        }
    return {
        "request_id": request_id,
        "answer": answer_provider.answer(payload.question, sources, all_context),
        "sources": [{"protocol": source.protocol, "page": source.page, "text": source.text} for source in sources],
        "knowledge_sources": all_provenance,
        "mode": "mock",
    }


@app.post("/protocols/upload")
def upload_protocol(file: UploadFile = File(...), _: None = Depends(require_api_key)) -> dict:
    content = file.file.read()
    filename = Path(file.filename or "protocol.pdf").name
    suffix = Path(filename).suffix.lower()

    if suffix == ".json":
        try:
            uploaded_data = json.loads(content.decode("utf-8"))
            protocols_to_add = uploaded_data["protocols"]
            if not isinstance(protocols_to_add, list) or not protocols_to_add:
                raise ValueError("protocols must be a non-empty list")
            for protocol in protocols_to_add:
                if not protocol.get("name") or not isinstance(protocol.get("chunks"), list):
                    raise ValueError("each protocol needs a name and chunks")
                for chunk in protocol["chunks"]:
                    if not chunk.get("page") or not chunk.get("text"):
                        raise ValueError("each chunk needs a page and text")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail="JSON must contain protocols with name and chunks.") from error

        store.replace(protocols_to_add, filename)
        refresh_retriever()
        return {"filename": filename, "protocols": len(protocols_to_add), "pages": sum(protocol.get("pages", 0) for protocol in protocols_to_add), "chunks": sum(len(protocol["chunks"]) for protocol in protocols_to_add), "status": "indexed", "storage": "sqlite"}

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF and JSON protocol files are supported.")

    try:
        reader = PdfReader(io.BytesIO(content))
        chunks = [{"page": index, "text": page.extract_text() or ""} for index, page in enumerate(reader.pages, start=1)]
    except Exception as error:
        raise HTTPException(status_code=400, detail="The PDF could not be read.") from error

    protocol_name = Path(filename).stem.replace("_", " ").strip() or "Uploaded protocol"
    protocol = {"name": protocol_name, "category": "Uploaded document", "version": "1.0", "updated": "Today", "pages": len(reader.pages), "chunks": [chunk for chunk in chunks if chunk["text"].strip()]}
    store.replace([protocol], filename)
    refresh_retriever()
    return {"filename": filename, "protocols": 1, "pages": len(reader.pages), "chunks": len(protocol["chunks"]), "status": "indexed", "storage": "sqlite"}
