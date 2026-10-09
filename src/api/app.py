"""
Production entry point for the Phase 5 streaming API.

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Configuration (environment variables):
    OLLAMA_HOST    default "localhost"
    OLLAMA_PORT    default "11434"
    OLLAMA_MODEL   default "mistral"
"""

import os

from src.api.streaming_server import create_app
from src.guardrails.judge import JudgeLLM
from src.guardrails.ollama_client import OllamaClient, OllamaConfig
from src.guardrails.pii_detector import PIIDetector
from src.guardrails.pii_masker import PIIMasker
from src.guardrails.stream_pipeline import PipelineConfig, StreamingGuardPipeline

SUPPORT_PROMPT = (
    "You are a customer support assistant. Answer the customer concisely.\n"
    "Customer: {query}\n"
    "Assistant:"
)


def create_default_app():
    host = os.getenv("OLLAMA_HOST", "localhost")
    port = int(os.getenv("OLLAMA_PORT", "11434"))
    model = os.getenv("OLLAMA_MODEL", "mistral")

    ollama = OllamaClient(OllamaConfig(host=host, port=port, model_name=model))

    async def token_source(query: str):
        async for fragment in ollama.stream_generate(SUPPORT_PROMPT.format(query=query)):
            yield fragment

    pipeline = StreamingGuardPipeline(
        detector=PIIDetector(),
        masker=PIIMasker(),
        judge=JudgeLLM(model_name=model, ollama_url=f"http://{host}:{port}"),
        config=PipelineConfig(),
    )
    return create_app(token_source, pipeline)


app = create_default_app()
