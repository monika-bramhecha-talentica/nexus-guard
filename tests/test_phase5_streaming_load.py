"""
Phase 5 (FR-5.7, PRD acceptance): Streaming load test over real sockets.

Runs a real uvicorn server on localhost and drives it with concurrent
WebSocket clients (websockets library). Token generation is simulated at
~50 tokens/second; Ollama is not required.

What this measures:
  - client-observed first-token latency (query sent -> first masked token received)
  - server-reported masking overhead per token (from the "done" metrics)
  - output integrity: no PII in any client's output, identical output across clients

Target (PRD): 10 concurrent users, P95 first-token latency < 200ms.
"""

import asyncio
import math
import socket
import threading
import time

import pytest
import uvicorn
import websockets

from src.api.streaming_server import SessionRegistry, create_app
from tests.test_phase5_streaming_pipeline import CANARY, make_pipeline

TOKEN_INTERVAL_S = 0.02  # ~50 tokens/second simulated model speed
TOKENS = (
    ["Thanks for contacting us. "]
    + ["Your Aadhaar ", "4111 ", "1111 ", "1111 ", "is on file. "]
    + [f"Your request {i} is being processed normally. " for i in range(50)]
    + ["Anything else I can help with?"]
)
PROMPT = "I need help with my account"


async def simulated_model(query):
    for token in TOKENS:
        await asyncio.sleep(TOKEN_INTERVAL_S)
        yield token


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _ServerThread:
    """Runs uvicorn in a background thread for the duration of a test."""

    def __init__(self):
        self.port = _free_port()
        app = create_app(
            lambda query: simulated_model(query),
            make_pipeline(flush_interval_ms=50, flush_tokens=20, holdback_words=3),
            registry=SessionRegistry(),
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", ws="websockets")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/stream"


async def run_client(url: str, query: str):
    """One user: send query, time first token, read to done."""
    async with websockets.connect(url, max_size=2**20) as ws:
        sent_at = time.perf_counter()
        await ws.send(_json({"type": "query", "query": query}))
        first_token_ms = None
        parts = []
        done = None
        while True:
            frame = await asyncio.wait_for(ws.recv(), timeout=30)
            event = _parse(frame)
            if event["type"] == "token":
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - sent_at) * 1000
                parts.append(event["text"])
            elif event["type"] == "error":
                raise AssertionError(f"server error: {event}")
            elif event["type"] == "done":
                done = event
                break
    return {
        "first_token_ms": first_token_ms,
        "text": "".join(parts),
        "metrics": done["metrics"],
        "final": None,
    }


def _json(obj):
    import json

    return json.dumps(obj)


def _parse(frame):
    import json

    return json.loads(frame)


def percentile(values, pct):
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


async def run_concurrent(url: str, users: int):
    return await asyncio.gather(*[run_client(url, PROMPT) for _ in range(users)])


class TestStreamingLoad:
    def test_ten_concurrent_users_meet_first_token_p95_target(self):
        with _ServerThread() as server:
            results = asyncio.run(run_concurrent(server.url, users=10))

        assert len(results) == 10
        first_token = [r["first_token_ms"] for r in results]
        p95 = percentile(first_token, 95)
        print(f"\n10 users - first-token ms: min={min(first_token):.1f} "
              f"p95={p95:.1f} max={max(first_token):.1f}")
        assert p95 < 200, f"P95 first-token latency {p95:.1f}ms exceeds 200ms target"

    def test_ten_concurrent_users_meet_masking_overhead_target(self):
        with _ServerThread() as server:
            results = asyncio.run(run_concurrent(server.url, users=10))

        per_token = [r["metrics"]["masking_overhead_per_token_ms"] for r in results]
        p95 = percentile(per_token, 95)
        print(f"\n10 users - masking overhead per token ms: p95={p95:.4f}")
        assert p95 < 50, f"P95 masking overhead {p95:.3f}ms/token exceeds 50ms target"

    def test_no_pii_leaks_and_output_identical_across_users(self):
        with _ServerThread() as server:
            results = asyncio.run(run_concurrent(server.url, users=10))

        texts = [r["text"] for r in results]
        for text in texts:
            assert CANARY not in text
            assert "MASKED" in text
        assert len(set(texts)) == 1, "identical input must produce identical masked output"

    def test_every_stream_completes_with_all_tokens_delivered(self):
        with _ServerThread() as server:
            results = asyncio.run(run_concurrent(server.url, users=10))

        expected_tokens = len(TOKENS)
        for result in results:
            assert result["metrics"]["tokens_in"] == expected_tokens
            assert result["text"].endswith("Anything else I can help with?")

    def test_twenty_five_concurrent_users_all_complete(self):
        """Headroom check beyond the PRD's 10-user target."""
        with _ServerThread() as server:
            results = asyncio.run(run_concurrent(server.url, users=25))

        assert len(results) == 25
        assert all(r["metrics"]["token_events"] >= 1 for r in results)
        assert all(CANARY not in r["text"] for r in results)
