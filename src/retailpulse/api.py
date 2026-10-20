"""HTTP API for the RetailPulse agent.

    uv run uvicorn retailpulse.api:app --port 8000

POST /ask            {question, thread_id?}   -> completed answer, or approval_required
POST /approve        {thread_id, approve}     -> resumes a paused question
POST /ask/stream     same body, Server-Sent Events: tool_call, tool_result, then answer | approval_required
POST /approve/stream same as /approve, streamed
GET  /health

Human-in-the-loop is a round trip: when a query needs approval the run is checkpointed by LangGraph and
the response says so; the client shows the SQL to a person and calls /approve. Reusing a thread_id asks
a follow-up question in the same conversation. (Checkpoints live in memory: one process, no restarts.
Production would use a Postgres/Redis checkpointer so any replica can resume any thread.)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from retailpulse.agent.graph import RetailAgent
from retailpulse.config import settings


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=2000, examples=["Which store had the highest revenue in 2015?"]
    )
    thread_id: str | None = Field(None, description="Reuse to ask a follow-up in the same conversation")


class ApproveRequest(BaseModel):
    thread_id: str
    approve: bool


def create_app(agent: RetailAgent | None = None) -> FastAPI:
    state: dict = {"agent": agent}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["agent"] is None:
            from retailpulse.llm import get_llm
            from retailpulse.observability import get_observer
            from retailpulse.tools.registry import Toolbox

            state["agent"] = RetailAgent(
                get_llm(),
                Toolbox.from_settings(),
                max_steps=settings.agent_max_steps,
                observer=get_observer(),
            )
        yield

    app = FastAPI(title="RetailPulse Agent API", version="0.1.0", lifespan=lifespan)

    def _agent() -> RetailAgent:
        return state["agent"]

    def _call(fn, *args, **kwargs) -> dict:
        try:
            return fn(*args, **kwargs).summary()
        except ValueError as e:  # thread busy with a pending approval
            raise HTTPException(409, str(e)) from e
        except KeyError as e:  # nothing to resume
            raise HTTPException(404, str(e.args[0])) from e

    def _sse(events: Iterator[dict]) -> EventSourceResponse:
        async def gen():
            async for e in iterate_in_threadpool(events):
                yield {"event": e["type"], "data": json.dumps(e, default=str)}

        return EventSourceResponse(gen())

    @app.get("/health")
    def health() -> dict:
        a = _agent()
        return {"status": "ok", "llm_provider": settings.llm_provider, "model": getattr(a.llm, "model", None)}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        return _call(_agent().start, req.question, thread_id=req.thread_id)

    @app.post("/approve")
    def approve(req: ApproveRequest) -> dict:
        return _call(_agent().resume, req.thread_id, req.approve)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> EventSourceResponse:
        return _sse(_agent().stream(req.question, thread_id=req.thread_id))

    @app.post("/approve/stream")
    def approve_stream(req: ApproveRequest) -> EventSourceResponse:
        if req.thread_id not in _agent()._turns:
            raise HTTPException(404, f"No question awaiting approval on thread {req.thread_id}.")
        return _sse(_agent().stream_resume(req.thread_id, req.approve))

    return app


app = create_app()
