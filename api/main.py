# api/main.py
"""
Ather EV Assistant — FastAPI Backend

Endpoints:
  POST /query      → full RAG pipeline, returns answer + metadata
  POST /feedback   → store thumbs up/down for a logged query
  GET  /analytics  → aggregated stats for the dashboard
  GET  /health     → system status check

All request/response shapes are Pydantic models — no raw dicts.

Startup sequence (runs once when uvicorn starts):
  1. init_db()           — create SQLite table if needed
  2. get_hybrid_searcher() — load BGE embedder + cross-encoder + ChromaDB
  3. get_prompt_chain()    — initialise Gemini client + memory store

These are singletons — heavy models are loaded once and reused.
"""

import time
import uuid
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Internal modules ──────────────────────────────────────────────────────────
from src.db.feedback_store        import init_db, log_query, update_rating, get_analytics
from src.query.intent_classifier  import classify_intent, intent_to_content_filter
from src.query.query_rewriter     import rewrite_query
from src.query.memory             import get_memory
from src.retrieval.hybrid_search  import get_hybrid_searcher, hybrid_search
from src.generation.prompt_chain  import get_prompt_chain
from src.generation.prompts       import OUT_OF_SCOPE_RESPONSE


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup: load all heavy singletons so the first request
    is not slow. Everything is cached after the first load.
    """
    logger.info("Ather EV Assistant API starting up...")

    init_db()
    logger.info("SQLite feedback DB ready")

    get_hybrid_searcher()
    logger.info("Hybrid searcher ready (BGE + cross-encoder + ChromaDB + BM25)")

    get_prompt_chain()
    logger.info("Prompt chain ready (Gemini client + memory)")

    logger.info("API ready — all systems operational")

    yield   # ← application runs here

    logger.info("API shutting down")


# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Ather EV Assistant API",
    description = "RAG-based chatbot for Ather electric scooter owners",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow Streamlit frontend (runs on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=500, description="User question")
    session_id: str = Field(default="", description="Session ID for conversation memory")

    model_config = {"json_schema_extra": {
        "example": {
            "query":      "how do I use the Ather Duo charger?",
            "session_id": "user-abc-123"
        }
    }}


class ChunkMetadata(BaseModel):
    page_number:  int | None  = None
    section:      str | None  = None
    content_type: str | None  = None
    vehicle_model: str | None = None
    source_file:  str | None  = None


class QueryResponse(BaseModel):
    answer:         str
    intent:         str
    rewritten_query: str
    cited_pages:    list[int]
    fallback_used:  bool
    low_confidence: bool
    latency_ms:     int
    session_id:     str
    row_id:         int         # use this to submit feedback


class FeedbackRequest(BaseModel):
    row_id: int   = Field(..., description="row_id from QueryResponse")
    rating: int   = Field(..., ge=-1, le=1, description="1=thumbs up, -1=thumbs down, 0=neutral")

    model_config = {"json_schema_extra": {
        "example": {"row_id": 42, "rating": 1}
    }}


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class HealthResponse(BaseModel):
    status:          str
    chroma_docs:     int
    model_locked:    str | None
    feedback_db:     str


class AnalyticsResponse(BaseModel):
    total_queries:  int
    thumbs_up:      int
    thumbs_down:    int
    fallback_count: int
    avg_latency_ms: float
    intent_counts:  dict[str, int]
    recent:         list[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Main RAG pipeline endpoint.

    Flow:
      1. Assign session_id if not provided
      2. Classify intent → skip RAG if out_of_scope
      3. Rewrite query for retrieval
      4. Hybrid search (dense + BM25 + reranker)
      5. 3-step generation (sufficiency → generate → validate)
      6. Log to SQLite
      7. Return response
    """
    request_start = time.time()

    # ── Session ID ────────────────────────────────────────────────────────
    session_id = request.session_id.strip() or str(uuid.uuid4())
    query      = request.query.strip()

    logger.info(f"[{session_id}] Query: '{query[:80]}'")

    # ── Step 1: Intent classification ─────────────────────────────────────
    intent_result  = classify_intent(query)
    intent         = intent_result["intent"]
    content_filter = intent_to_content_filter(intent)

    # ── Step 2: Out-of-scope short circuit ────────────────────────────────
    if intent_result["skip_rag"]:
        logger.info(f"[{session_id}] Out of scope — skipping RAG")

        row_id = log_query(
            session_id  = session_id,
            query       = query,
            rewritten   = query,
            intent      = intent,
            response    = OUT_OF_SCOPE_RESPONSE,
            fallback    = True,
            latency_ms  = int((time.time() - request_start) * 1000),
            cited_pages = [],
        )

        return QueryResponse(
            answer          = OUT_OF_SCOPE_RESPONSE,
            intent          = intent,
            rewritten_query = query,
            cited_pages     = [],
            fallback_used   = True,
            low_confidence  = False,
            latency_ms      = int((time.time() - request_start) * 1000),
            session_id      = session_id,
            row_id          = row_id,
        )

    # ── Step 3: Query rewriting ───────────────────────────────────────────
    memory         = get_memory()
    previous_query = memory.get_last_query(session_id)
    rewrite_result = rewrite_query(query, previous_query=previous_query)
    rewritten      = rewrite_result["rewritten"]

    logger.info(f"[{session_id}] Rewritten: '{rewritten[:80]}'")

    # ── Step 4: Hybrid retrieval ──────────────────────────────────────────
    retrieval_result = hybrid_search(
        query               = rewritten,
        content_type_filter = content_filter,
    )

    # ── Step 5: Generation ────────────────────────────────────────────────
    chain  = get_prompt_chain()
    result = chain.run(
        query            = rewritten,
        retrieval_result = retrieval_result,
        session_id       = session_id,
    )

    total_latency = int((time.time() - request_start) * 1000)

    # ── Step 6: Log to SQLite ─────────────────────────────────────────────
    row_id = log_query(
        session_id  = session_id,
        query       = query,
        rewritten   = rewritten,
        intent      = intent,
        response    = result["answer"],
        fallback    = result["fallback_used"],
        latency_ms  = total_latency,
        cited_pages = result["cited_pages"],
    )

    logger.info(
        f"[{session_id}] Done | intent={intent} | "
        f"fallback={result['fallback_used']} | "
        f"latency={total_latency}ms | row_id={row_id}"
    )

    # ── Step 7: Return ────────────────────────────────────────────────────
    return QueryResponse(
        answer          = result["answer"],
        intent          = intent,
        rewritten_query = rewritten,
        cited_pages     = result["cited_pages"],
        fallback_used   = result["fallback_used"],
        low_confidence  = result["low_confidence"],
        latency_ms      = total_latency,
        session_id      = session_id,
        row_id          = row_id,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(request: FeedbackRequest) -> FeedbackResponse:
    """
    Store a thumbs up/down rating for a previously answered query.
    The row_id comes from the QueryResponse of the original query call.
    """
    success = update_rating(request.row_id, request.rating)

    if not success:
        raise HTTPException(
            status_code = 404,
            detail      = f"row_id {request.row_id} not found in feedback store"
        )

    rating_label = {1: "thumbs up", -1: "thumbs down", 0: "neutral"}[request.rating]
    logger.info(f"Feedback: row_id={request.row_id} → {rating_label}")

    return FeedbackResponse(
        success = True,
        message = f"Rating '{rating_label}' recorded for query #{request.row_id}"
    )


@app.get("/analytics", response_model=AnalyticsResponse)
async def analytics_endpoint() -> AnalyticsResponse:
    """
    Return aggregated stats for the Streamlit analytics dashboard.
    """
    stats = get_analytics()
    return AnalyticsResponse(**stats)


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """
    System health check. Verifies ChromaDB, Gemini client, and SQLite are live.
    """
    from src.retrieval.dense_search   import get_dense_searcher
    from src.generation.gemini_client import _active_model

    try:
        searcher   = get_dense_searcher()
        chroma_docs = searcher.collection.count()
        chroma_ok   = True
    except Exception as e:
        logger.error(f"Health check — ChromaDB error: {e}")
        chroma_docs = -1
        chroma_ok   = False

    try:
        from src.db.feedback_store import _get_connection
        conn = _get_connection()
        conn.execute("SELECT 1 FROM feedback LIMIT 1")
        conn.close()
        db_status = "ok"
    except Exception:
        db_status = "error"

    overall = "ok" if (chroma_ok and db_status == "ok") else "degraded"

    return HealthResponse(
        status       = overall,
        chroma_docs  = chroma_docs,
        model_locked = _active_model,
        feedback_db  = db_status,
    )


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = False,    # reload=True breaks singleton model loading
        log_level= "info",
    )