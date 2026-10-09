"""
Phase 5 (FR-5.1, FR-5.8): WebSocket streaming API.

Protocol (JSON text frames):
  client -> {"type": "query", "query": "..."}              start a new stream
  client -> {"type": "resume", "stream_id": "...", "from_seq": N}
                                                           replay events with seq > N, then follow live
  server -> {"seq": 1, "type": "started", ...}
  server -> {"seq": k, "type": "token", "text": "..."}      masked output fragment
  server -> {"seq": k, "type": "status", "state": "judging"}
  server -> {"seq": k, "type": "final", "status": "approved|escalated|held", ...}
  server -> {"seq": k, "type": "done", "metrics": {...}}
  server -> {"type": "error", "code": "...", "message": "..."}  (not part of the seq log)

The pipeline runs as a background task owned by the server, not by the socket.
If the client disconnects, the stream keeps running and its events stay in the
session log until the registry expires them, so the client can resume.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.guardrails.stream_pipeline import StreamingGuardPipeline, StreamSession

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 4000
DEFAULT_SESSION_TTL_SECONDS = 300.0

TokenSourceFactory = Callable[[str], AsyncIterator[str]]


class SessionRegistry:
    """In-memory registry of stream sessions with TTL cleanup after completion."""

    def __init__(self, ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, StreamSession] = {}

    def create(self) -> StreamSession:
        stream_id = uuid.uuid4().hex
        session = StreamSession(stream_id)
        self._sessions[stream_id] = session
        return session

    def get(self, stream_id: str) -> Optional[StreamSession]:
        return self._sessions.get(stream_id)

    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.finished)

    def gc(self, now: Optional[float] = None) -> int:
        """Drop finished sessions older than the TTL. Returns number removed."""
        now = time.monotonic() if now is None else now
        expired = [
            sid
            for sid, s in self._sessions.items()
            if s.finished and s.finished_at is not None and now - s.finished_at > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


def create_app(
    token_source_factory: TokenSourceFactory,
    pipeline: StreamingGuardPipeline,
    registry: Optional[SessionRegistry] = None,
) -> FastAPI:
    """Build the FastAPI application serving the streaming WebSocket."""
    registry = registry or SessionRegistry()
    app = FastAPI(title="Nexus Guard Streaming API")
    app.state.registry = registry

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        registry.gc()
        return {"status": "ok", "active_streams": registry.active_count()}

    @app.websocket("/ws/stream")
    async def stream_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            message = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except ValueError:
            await _send_error(websocket, "bad_json", "First frame must be JSON.")
            await websocket.close(code=1003)
            return

        msg_type = message.get("type")

        if msg_type == "query":
            query = str(message.get("query", "")).strip()
            agent_id = str(message.get("agent_id", "agent_unknown"))
            if not query:
                await _send_error(websocket, "empty_query", "Query must not be empty.")
                await websocket.close(code=1008)
                return
            if len(query) > MAX_QUERY_CHARS:
                await _send_error(websocket, "query_too_long", f"Max {MAX_QUERY_CHARS} characters.")
                await websocket.close(code=1009)
                return
            session = registry.create()
            session.task = asyncio.create_task(
                pipeline.run(
                    session,
                    token_source_factory(query),
                    query=query,
                    agent_id=agent_id,
                )
            )
            from_seq = 0

        elif msg_type == "resume":
            session = registry.get(str(message.get("stream_id", "")))
            if session is None:
                await _send_error(websocket, "unknown_stream", "Stream not found or expired.")
                await websocket.close(code=1008)
                return
            try:
                from_seq = max(0, int(message.get("from_seq", 0)))
            except (TypeError, ValueError):
                await _send_error(websocket, "bad_from_seq", "from_seq must be an integer.")
                await websocket.close(code=1003)
                return

        else:
            await _send_error(websocket, "bad_type", "Expected type 'query' or 'resume'.")
            await websocket.close(code=1003)
            return

        try:
            async for event in session.subscribe(from_seq):
                await websocket.send_json(event)
                if event["type"] in ("done", "error"):
                    break
        except WebSocketDisconnect:
            logger.info("Client disconnected from stream %s; stream keeps running", session.stream_id)
            return
        await websocket.close(code=1000)

    return app


async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "code": code, "message": message})
    except WebSocketDisconnect:
        pass
