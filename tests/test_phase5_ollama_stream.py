"""
Phase 5 (FR-5.2): Ollama token streaming client.

Uses a local aiohttp server that speaks Ollama's /api/generate NDJSON
protocol, so the streaming parser is exercised without a real Ollama install.
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from src.guardrails.ollama_client import OllamaClient, OllamaConfig


def ndjson_handler_factory(fragments, fail_status=None):
    async def handler(request):
        body = await request.json()
        assert body["stream"] is True
        if fail_status:
            return web.Response(status=fail_status, text="model not found")
        response = web.StreamResponse(status=200, headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        for fragment in fragments:
            line = json.dumps({"model": body["model"], "response": fragment, "done": False}) + "\n"
            await response.write(line.encode())
        await response.write(
            (json.dumps({"model": body["model"], "response": "", "done": True}) + "\n").encode()
        )
        await response.write_eof()
        return response

    return handler


async def make_server(handler):
    app = web.Application()
    app.router.add_post("/api/generate", handler)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.asyncio
async def test_stream_generate_yields_fragments_in_order():
    fragments = ["Hello", " there", ", how", " can I help?"]
    server = await make_server(ndjson_handler_factory(fragments))
    try:
        client = OllamaClient(
            OllamaConfig(host="127.0.0.1", port=server.port, model_name="test-model")
        )
        received = [f async for f in client.stream_generate("hi")]
    finally:
        await server.close()
    assert received == fragments


@pytest.mark.asyncio
async def test_stream_generate_skips_empty_fragments_and_stops_on_done():
    fragments = ["A", "", "B"]
    server = await make_server(ndjson_handler_factory(fragments))
    try:
        client = OllamaClient(OllamaConfig(host="127.0.0.1", port=server.port))
        received = [f async for f in client.stream_generate("hi")]
    finally:
        await server.close()
    assert received == ["A", "B"]


@pytest.mark.asyncio
async def test_stream_generate_raises_on_http_error():
    server = await make_server(ndjson_handler_factory([], fail_status=404))
    try:
        client = OllamaClient(OllamaConfig(host="127.0.0.1", port=server.port))
        with pytest.raises(Exception):
            _ = [f async for f in client.stream_generate("hi")]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_stream_generate_sends_configured_model():
    seen = {}

    async def handler(request):
        body = await request.json()
        seen["model"] = body["model"]
        response = web.StreamResponse(status=200)
        await response.prepare(request)
        await response.write(b'{"response":"ok","done":true}\n')
        await response.write_eof()
        return response

    server = await make_server(handler)
    try:
        client = OllamaClient(
            OllamaConfig(host="127.0.0.1", port=server.port, model_name="mistral")
        )
        _ = [f async for f in client.stream_generate("hi")]
    finally:
        await server.close()
    assert seen["model"] == "mistral"
