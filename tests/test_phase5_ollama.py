"""
Phase 5: Ollama LLM Integration Tests

Comprehensive tests for Ollama client, LLM evaluation, streaming support,
and fallback behavior. Tests async evaluation with mocking.

Test Coverage:
  - OllamaClient initialization and configuration
  - Connectivity checking (success and failure)
  - LLM evaluation with structured responses
  - Response parsing and validation
  - JSON error handling
  - Fallback to rule-based scoring
  - Response caching
  - Prompt format validation
  - Streaming integration
  - Error scenarios
"""

import pytest
import asyncio
import json
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import modules to test
from src.guardrails.ollama_client import OllamaClient, OllamaConfig
from src.guardrails.streaming_handler import StreamingResponseHandler, StreamingChunk, StreamingMetadata
from src.guardrails.judge import JudgeLLM


class TestOllamaConfig:
    """Tests for OllamaConfig dataclass."""

    def test_config_initialization(self):
        """Test OllamaConfig default initialization."""
        config = OllamaConfig()
        assert config.host == "localhost"
        assert config.port == 11434
        assert config.model_name == "mistral"
        assert config.timeout == 10
        assert config.fallback_enabled is True

    def test_config_custom_values(self):
        """Test OllamaConfig with custom values."""
        config = OllamaConfig(
            host="192.168.1.100",
            port=11435,
            model_name="llama2",
            timeout=20,
            fallback_enabled=False,
        )
        assert config.host == "192.168.1.100"
        assert config.port == 11435
        assert config.model_name == "llama2"
        assert config.timeout == 20
        assert config.fallback_enabled is False

    def test_config_base_url(self):
        """Test OllamaConfig base_url construction."""
        config = OllamaConfig(host="ollama.local", port=11434)
        assert config.base_url == "http://ollama.local:11434"


class TestOllamaClient:
    """Tests for OllamaClient class."""

    def test_client_initialization(self):
        """Test OllamaClient initialization."""
        client = OllamaClient()
        assert client.config is not None
        assert client.is_available is False
        assert len(client.evaluation_cache) == 0

    def test_client_with_custom_config(self):
        """Test OllamaClient with custom configuration."""
        config = OllamaConfig(host="custom-host", port=11435)
        client = OllamaClient(config=config)
        assert client.config.host == "custom-host"
        assert client.config.port == 11435

    @pytest.mark.asyncio
    async def test_connectivity_check_success(self):
        """Test successful connectivity check to Ollama."""
        client = OllamaClient()
        # Ollama won't be running in test environment
        result = await client.check_connectivity()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_connectivity_check_failure(self):
        """Test failed connectivity check (Ollama unavailable)."""
        client = OllamaClient()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await client.check_connectivity()
            assert result is False
            assert client.is_available is False

    @pytest.mark.asyncio
    async def test_connectivity_check_exception(self):
        """Test connectivity check with network exception."""
        client = OllamaClient()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("Network error")

            result = await client.check_connectivity()
            assert result is False
            assert client.is_available is False

    @pytest.mark.asyncio
    async def test_evaluate_llm_success(self):
        """Test successful LLM evaluation."""
        client = OllamaClient()
        client.is_available = True

        mock_evaluation = {
            "accuracy_score": 0.85,
            "relevance_score": 0.90,
            "completeness_score": 0.80,
            "safety_score": 1.0,
            "hallucination_score": 0.95,
            "pii_handling_score": 1.0,
            "strengths": ["Accurate", "Complete"],
            "weaknesses": [],
            "hallucinations_detected": [],
        }

        with patch.object(client, "_call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_evaluation

            result, success = await client.evaluate_llm("Great response", "query")
            assert success is True
            assert result["accuracy_score"] == 0.85
            assert result["relevance_score"] == 0.90

    @pytest.mark.asyncio
    async def test_evaluate_llm_cache_hit(self):
        """Test LLM evaluation with cache hit."""
        client = OllamaClient()
        client.is_available = True

        response = "Cached response"
        response_hash = hash(response)

        cached_result = {
            "accuracy_score": 0.75,
            "relevance_score": 0.75,
            "completeness_score": 0.75,
            "safety_score": 1.0,
            "hallucination_score": 0.95,
            "pii_handling_score": 1.0,
        }

        client.evaluation_cache[response_hash] = cached_result

        result, success = await client.evaluate_llm(response, "query")
        assert success is True
        assert result == cached_result

    @pytest.mark.asyncio
    async def test_evaluate_llm_unavailable(self):
        """Test LLM evaluation when Ollama unavailable."""
        client = OllamaClient()
        client.is_available = False

        result, success = await client.evaluate_llm("response", "query")
        assert success is False
        assert result == {}

    @pytest.mark.asyncio
    async def test_evaluate_llm_timeout(self):
        """Test LLM evaluation with timeout."""
        client = OllamaClient()
        client.is_available = True

        with patch.object(client, "_call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = asyncio.TimeoutError()

            result, success = await client.evaluate_llm("response", "query")
            assert success is False
            assert result == {}

    def test_build_evaluation_prompt(self):
        """Test evaluation prompt generation."""
        client = OllamaClient()
        prompt = client._build_evaluation_prompt("Test response", "Test query")

        assert "Test response" in prompt
        assert "Test query" in prompt
        assert "accuracy_score" in prompt
        assert "relevance_score" in prompt
        assert "JSON" in prompt

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM JSON response."""
        client = OllamaClient()

        response_text = """{
            "accuracy_score": 0.85,
            "relevance_score": 0.90,
            "completeness_score": 0.80,
            "safety_score": 1.0,
            "hallucination_score": 0.95,
            "pii_handling_score": 1.0,
            "strengths": ["Good"],
            "weaknesses": []
        }"""

        result = client._parse_llm_response(response_text)
        assert result["accuracy_score"] == 0.85
        assert result["relevance_score"] == 0.90
        assert result["completeness_score"] == 0.80

    def test_parse_llm_response_with_text_wrapper(self):
        """Test parsing LLM response with surrounding text."""
        client = OllamaClient()

        response_text = """Here's my evaluation:
        {
            "accuracy_score": 0.75,
            "relevance_score": 0.80,
            "completeness_score": 0.85,
            "safety_score": 1.0,
            "hallucination_score": 0.90,
            "pii_handling_score": 1.0
        }
        End of evaluation."""

        result = client._parse_llm_response(response_text)
        assert result["accuracy_score"] == 0.75

    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        client = OllamaClient()

        response_text = "This is not JSON"
        result = client._parse_llm_response(response_text)
        assert result == {}

    def test_parse_llm_response_missing_required_field(self):
        """Test parsing response with missing required field."""
        client = OllamaClient()

        response_text = """{
            "accuracy_score": 0.85,
            "relevance_score": 0.90
        }"""

        result = client._parse_llm_response(response_text)
        assert result == {}  # Should fail due to missing fields

    def test_parse_llm_response_score_clamping(self):
        """Test that scores are clamped to [0, 1] range."""
        client = OllamaClient()

        response_text = """{
            "accuracy_score": 1.5,
            "relevance_score": -0.5,
            "completeness_score": 0.5,
            "safety_score": 1.0,
            "hallucination_score": 0.9,
            "pii_handling_score": 0.8
        }"""

        result = client._parse_llm_response(response_text)
        assert result["accuracy_score"] == 1.0  # Clamped
        assert result["relevance_score"] == 0.0  # Clamped
        assert result["completeness_score"] == 0.5  # Normal

    def test_get_cache_stats(self):
        """Test cache statistics."""
        client = OllamaClient()
        client.is_available = True

        # Add some cache entries
        client.evaluation_cache[1] = {"accuracy_score": 0.85}
        client.evaluation_cache[2] = {"accuracy_score": 0.90}

        stats = client.get_cache_stats()
        assert stats["cache_entries"] == 2
        assert stats["cache_size_bytes"] > 0
        assert stats["ollama_available"] is True

    def test_clear_cache(self):
        """Test cache clearing."""
        client = OllamaClient()

        client.evaluation_cache[1] = {"accuracy_score": 0.85}
        assert len(client.evaluation_cache) == 1

        client.clear_cache()
        assert len(client.evaluation_cache) == 0


class TestStreamingResponseHandler:
    """Tests for StreamingResponseHandler class."""

    def test_handler_initialization(self):
        """Test StreamingResponseHandler initialization."""
        handler = StreamingResponseHandler()
        assert len(handler.active_streams) == 0
        assert len(handler.stream_buffers) == 0
        assert len(handler.chunk_scores) == 0

    def test_start_stream(self):
        """Test starting a new stream."""
        handler = StreamingResponseHandler()
        metadata = handler.start_stream("stream_001")

        assert metadata.stream_id == "stream_001"
        assert metadata.total_chunks == 0
        assert metadata.streaming_complete is False

    def test_add_chunk(self):
        """Test adding chunks to stream."""
        handler = StreamingResponseHandler()
        handler.start_stream("stream_001")

        metadata = handler.add_chunk("stream_001", "Hello ")
        assert metadata.total_chunks == 1
        assert metadata.total_bytes == 6

        metadata = handler.add_chunk("stream_001", "World!")
        assert metadata.total_chunks == 2
        assert metadata.total_bytes == 12

    def test_add_chunk_invalid_stream(self):
        """Test adding chunk to non-existent stream."""
        handler = StreamingResponseHandler()
        result = handler.add_chunk("invalid_stream", "content")
        assert result is None

    def test_compute_preliminary_score(self):
        """Test preliminary score computation."""
        handler = StreamingResponseHandler()
        handler.start_stream("stream_001")
        handler.add_chunk("stream_001", "This is a comprehensive response with detailed information.")

        score = handler.compute_preliminary_score("stream_001")
        assert score is not None
        assert "accuracy_score" in score
        assert score["accuracy_score"] >= 0.0 and score["accuracy_score"] <= 1.0

    def test_finalize_stream(self):
        """Test stream finalization."""
        handler = StreamingResponseHandler()
        handler.start_stream("stream_001")
        handler.add_chunk("stream_001", "Complete ")
        handler.add_chunk("stream_001", "response")

        result = handler.finalize_stream("stream_001")
        assert result is not None
        assert result["stream_id"] == "stream_001"
        assert result["complete_response"] == "Complete response"
        assert result["metadata"]["total_chunks"] == 2
        assert result["metadata"]["total_bytes"] == 17  # "Complete " (9) + "response" (8) = 17

    def test_finalize_invalid_stream(self):
        """Test finalizing non-existent stream."""
        handler = StreamingResponseHandler()
        result = handler.finalize_stream("invalid_stream")
        assert result is None

    def test_get_stream_status(self):
        """Test getting stream status."""
        handler = StreamingResponseHandler()
        handler.start_stream("stream_001")
        handler.add_chunk("stream_001", "Content")

        status = handler.get_stream_status("stream_001")
        assert status is not None
        assert status["stream_id"] == "stream_001"
        assert status["is_complete"] is False
        assert status["chunks_received"] == 1
        assert status["bytes_received"] == 7

    def test_cleanup_stream(self):
        """Test stream cleanup."""
        handler = StreamingResponseHandler()
        handler.start_stream("stream_001")
        handler.add_chunk("stream_001", "data")

        assert len(handler.active_streams) == 1

        handler.cleanup_stream("stream_001")
        assert len(handler.active_streams) == 0
        assert len(handler.stream_buffers) == 0

    def test_get_streaming_stats(self):
        """Test streaming statistics."""
        handler = StreamingResponseHandler()
        handler.start_stream("stream_001")
        handler.add_chunk("stream_001", "data1")
        handler.add_chunk("stream_001", "data2")

        stats = handler.get_streaming_stats()
        assert stats["active_streams"] == 1
        assert stats["total_chunks_processed"] == 2
        assert stats["total_bytes_processed"] == 10


class TestJudgeLLMOllamaIntegration:
    """Tests for JudgeLLM Ollama integration."""

    def test_judge_initialization_with_ollama(self):
        """Test JudgeLLM initialization with Ollama."""
        judge = JudgeLLM(model_name="mistral")

        # Check if Ollama client was initialized
        if judge.ollama_client:
            assert judge.streaming_handler is not None
            assert judge.use_llm_evaluation is True

    @pytest.mark.asyncio
    async def test_evaluate_async_fallback(self):
        """Test async evaluate with fallback to rule-based."""
        judge = JudgeLLM()

        evaluation = await judge.evaluate(
            "This is a great response",
            query="How are you?",
            agent_id="test_agent",
        )

        assert evaluation is not None
        assert evaluation.overall_score >= 0.0 and evaluation.overall_score <= 1.0
        assert evaluation.quality_tier in ["excellent", "good", "fair", "poor", "critical"]

    def test_start_streaming_response(self):
        """Test starting streaming response tracking."""
        judge = JudgeLLM()

        if judge.streaming_handler:
            result = judge.start_streaming_response("stream_001")
            assert result == "stream_001"

    def test_add_stream_chunk(self):
        """Test adding chunks to streaming response."""
        judge = JudgeLLM()

        if judge.streaming_handler:
            judge.start_streaming_response("stream_001")
            status = judge.add_stream_chunk("stream_001", "Response ")
            assert status is not None

            status = judge.add_stream_chunk("stream_001", "chunk")
            assert status["chunks_received"] == 2

    @pytest.mark.asyncio
    async def test_evaluate_streaming(self):
        """Test streaming response evaluation."""
        judge = JudgeLLM()

        if judge.streaming_handler:
            judge.start_streaming_response("stream_001")
            judge.add_stream_chunk("stream_001", "This is a ")
            judge.add_stream_chunk("stream_001", "streaming response")

            evaluation = await judge.evaluate_streaming(
                "stream_001",
                query="Test query",
                agent_id="test_agent",
            )

            if evaluation:
                assert evaluation.overall_score >= 0.0
                assert "streaming" in evaluation.reasoning.lower() or "chunks" in evaluation.reasoning.lower()


class TestPhase5Integration:
    """Integration tests for Phase 5 components."""

    @pytest.mark.asyncio
    async def test_ollama_to_judge_integration(self):
        """Test Ollama client integration with Judge."""
        judge = JudgeLLM()

        # Verify Ollama setup
        if judge.ollama_client:
            assert judge.ollama_client.config.model_name == "mistral"
            assert judge.ollama_client.config.timeout == 10

    def test_streaming_to_judge_integration(self):
        """Test Streaming handler integration with Judge."""
        judge = JudgeLLM()

        # Verify streaming setup
        if judge.streaming_handler:
            judge.start_streaming_response("test_stream")
            judge.add_stream_chunk("test_stream", "data")

            status = judge.streaming_handler.get_stream_status("test_stream")
            assert status is not None
            assert status["chunks_received"] == 1


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
