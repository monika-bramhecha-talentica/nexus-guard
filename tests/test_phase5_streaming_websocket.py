"""
Phase 5 (FR-5.1, FR-5.8): WebSocket streaming API tests.

Uses FastAPI's TestClient with real WebSocket frames. The token source is a
deterministic async generator so assertions are exact.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from src.api.streaming_server import MAX_QUERY_CHARS, SessionRegistry, create_app
from tests.test_phase5_streaming_pipeline import CANARY, make_pipeline

TOKENS = ["Your Aadhaar ", "4111 ", "1111 ", "1111 ", "is on file. ", "Thanks for asking."]


def fake_factory(tokens=TOKENS, delay=0.0):
    async def gen(query):
        for token in tokens:
            if delay:
                await asyncio.sleep(delay)
            yield token

    return lambda query: gen(query)


def build_client(factory=None, ttl=300.0):
    registry = SessionRegistry(ttl_seconds=ttl)
    app = create_app(
        factory or fake_factory(),
        make_pipeline(flush_interval_ms=20, holdback_words=2),
        registry=registry,
    )
    return TestClient(app), registry


def read_until_terminal(ws, limit=500):
    events = []
    for _ in range(limit):
        event = ws.receive_json()
        if event.get("type") == "error" and "seq" not in event:
            events.append(event)
            break
        events.append(event)
        if event["type"] in ("done", "error"):
            break
    return events


class TestQueryStream:
    def test_full_stream_is_masked_and_ends_with_done(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "query", "query": "my aadhaar?"})
            events = read_until_terminal(ws)

        assert events[0]["type"] == "started"
        assert events[-1]["type"] == "done"
        text = "".join(e["text"] for e in events if e["type"] == "token")
        assert CANARY not in text
        assert "MASKED" in text
        assert text.endswith("Thanks for asking.")

    def test_sequence_numbers_are_contiguous(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "query", "query": "q"})
            events = read_until_terminal(ws)
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))

    def test_final_verdict_arrives_before_done(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "query", "query": "q"})
            events = read_until_terminal(ws)
        kinds = [e["type"] for e in events]
        assert kinds.index("final") < kinds.index("done")
        final = next(e for e in events if e["type"] == "final")
        assert final["status"] in ("approved", "escalated", "held")

    def test_done_contains_latency_metrics(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "query", "query": "q"})
            events = read_until_terminal(ws)
        metrics = events[-1]["metrics"]
        assert metrics["first_token_ms"] is not None
        assert metrics["tokens_in"] == len(TOKENS)


class TestProtocolValidation:
    def test_empty_query_is_rejected(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "query", "query": "   "})
            event = ws.receive_json()
        assert event == {"type": "error", "code": "empty_query", "message": "Query must not be empty."}

    def test_overlong_query_is_rejected(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "query", "query": "x" * (MAX_QUERY_CHARS + 1)})
            event = ws.receive_json()
        assert event["code"] == "query_too_long"

    def test_unknown_frame_type_is_rejected(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "hello"})
            event = ws.receive_json()
        assert event["code"] == "bad_type"

    def test_non_json_first_frame_is_rejected(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_text("this is not json")
            event = ws.receive_json()
        assert event["code"] == "bad_json"

    def test_resume_unknown_stream_is_rejected(self):
        client, _ = build_client()
        with client, client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"type": "resume", "stream_id": "nope", "from_seq": 0})
            event = ws.receive_json()
        assert event["code"] == "unknown_stream"

    def test_resume_with_bad_from_seq_is_rejected(self):
        client, registry = build_client()
        with client:
            session = registry.create()
            session.finished = True
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "resume", "stream_id": session.stream_id, "from_seq": "abc"})
                event = ws.receive_json()
        assert event["code"] == "bad_from_seq"


class TestResumeAndReconnect:
    def test_resume_continues_from_last_seq_without_gaps_or_duplicates(self):
        client, _ = build_client(fake_factory(delay=0.01))
        with client:
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "query", "query": "q"})
                first_part = [ws.receive_json() for _ in range(3)]
            # Connection dropped after seq 3. The stream keeps running server-side.
            stream_id = first_part[0]["stream_id"]
            last_seq = first_part[-1]["seq"]

            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "resume", "stream_id": stream_id, "from_seq": last_seq})
                second_part = read_until_terminal(ws)

        seqs = [e["seq"] for e in first_part] + [e["seq"] for e in second_part]
        assert seqs == list(range(1, len(seqs) + 1))
        assert second_part[-1]["type"] == "done"

    def test_resume_from_zero_replays_full_log(self):
        client, _ = build_client()
        with client:
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "query", "query": "q"})
                original = read_until_terminal(ws)
            stream_id = original[0]["stream_id"]
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "resume", "stream_id": stream_id, "from_seq": 0})
                replayed = read_until_terminal(ws)
        assert replayed == original

    def test_client_disconnect_does_not_stop_the_stream(self):
        client, registry = build_client(fake_factory(delay=0.01))
        with client:
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "query", "query": "q"})
                stream_id = ws.receive_json()["stream_id"]
            # Socket is closed. The pipeline must still finish.
            session = registry.get(stream_id)
            deadline = time.monotonic() + 5
            while not session.finished and time.monotonic() < deadline:
                time.sleep(0.02)
        assert session.finished
        assert session.events[-1]["type"] == "done"


class TestRegistryAndHealth:
    def test_health_reports_status(self):
        client, _ = build_client()
        with client:
            body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["active_streams"] == 0

    def test_gc_removes_only_finished_sessions_past_ttl(self):
        registry = SessionRegistry(ttl_seconds=10)

        finished = registry.create()
        running = registry.create()
        finished.finished = True
        finished.finished_at = 100.0
        removed = registry.gc(now=200.0)
        assert removed == 1
        assert registry.get(finished.stream_id) is None
        assert registry.get(running.stream_id) is not None

    def test_gc_keeps_recently_finished_sessions(self):
        registry = SessionRegistry(ttl_seconds=10)
        session = registry.create()
        session.finished = True
        session.finished_at = 100.0
        assert registry.gc(now=105.0) == 0
        assert registry.get(session.stream_id) is not None
