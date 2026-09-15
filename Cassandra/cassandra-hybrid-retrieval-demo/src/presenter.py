"""FastAPI presenter for the three hybrid retrieval lanes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common import ensure_schema, wait_for_cassandra
from retrieval import (
    DEFAULT_QUERIES,
    HIGHLIGHT_TOKEN,
    MAX_LIMIT,
    retrieve_all,
)

PRESENTER_HTML = Path(__file__).resolve().parents[1] / "static" / "presenter.html"
LOGGER = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=MAX_LIMIT)
    model: Optional[str] = Field(default=None, max_length=10)
    model_year: Optional[int] = Field(default=None, ge=1900, le=2100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = wait_for_cassandra()
    ensure_schema(session)
    app.state.cassandra = session
    yield
    session.cluster.shutdown()


app = FastAPI(title="Cassandra Hybrid Retrieval Presenter", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PRESENTER_HTML)


@app.get("/api/presets")
def presets() -> dict[str, object]:
    return {
        "queries": DEFAULT_QUERIES,
        "highlight_token": HIGHLIGHT_TOKEN,
        "models": ["1500", "2500"],
        "model_years": [2024, 2023, 2022],
    }


@app.post("/api/query")
def query(payload: QueryRequest, request: Request) -> dict[str, object]:
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=422, detail="Query cannot be blank")
    try:
        result = retrieve_all(
            request.app.state.cassandra,
            query_text,
            payload.limit,
            model=payload.model,
            model_year=payload.model_year,
        )
    except Exception as exc:
        LOGGER.exception("Retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Retrieval failed; check the server logs.",
        ) from exc
    return result.to_dict()
