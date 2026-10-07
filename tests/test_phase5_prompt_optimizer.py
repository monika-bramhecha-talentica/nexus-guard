"""
Phase 5 Day 2: Prompt Optimizer Tests

Tests for prompt optimization and latency reduction:
  - Multiple prompt versions (V1-V4)
  - Token estimation
  - Prompt comparison
  - Compact format validation
  - Experiment tracking
"""

import pytest
from src.guardrails.prompt_optimizer import (
    PromptVersion,
    PromptOptimizer,
    PromptExperiment,
)


class TestPromptOptimizer:
    """Tests for prompt optimization."""

    def test_token_estimation(self):
        """Test token count estimation."""
        text = "This is a test"
        tokens = PromptOptimizer.estimate_tokens(text)

        # "This is a test" = 14 chars, should be ~3-4 tokens
        assert 2 < tokens < 5

    def test_token_estimation_accuracy(self):
        """Test token estimation on longer text."""
        # 400 chars should be ~100 tokens
        text = "x" * 400
        tokens = PromptOptimizer.estimate_tokens(text)

        assert 90 < tokens < 110

    def test_generate_v1_standard(self):
        """Test V1 standard prompt generation."""
        response = "This is a response"
        query = "This is a query"

        prompt = PromptOptimizer.generate_v1_standard(response, query)

        assert "accuracy_score" in prompt
        assert "relevance_score" in prompt
        assert "This is a response" in prompt
        assert "This is a query" in prompt

    def test_generate_v2_compact(self):
        """Test V2 compact prompt generation."""
        response = "Response text"
        query = "Query text"

        prompt = PromptOptimizer.generate_v2_compact(response, query)

        # Should be shorter than V1
        assert len(prompt) < 300
        assert "accuracy" in prompt
        assert "Response text" in prompt

    def test_generate_v3_ultra_compact(self):
        """Test V3 ultra-compact prompt generation."""
        response = "Response text"
        query = "Query text"

        prompt = PromptOptimizer.generate_v3_ultra_compact(response, query)

        # Should be very short
        assert len(prompt) < 200
        assert "accuracy" in prompt

    def test_generate_v4_selective(self):
        """Test V4 selective prompt generation."""
        response = "Response"
        query = "Query"

        # Skip certain dimensions
        prompt = PromptOptimizer.generate_v4_selective(
            response, query, skip_dimensions=["hallucination"]
        )

        assert "accuracy" in prompt
        assert "hallucination" not in prompt.split("{")[1]  # Not in JSON

    def test_generate_optimized_versions(self):
        """Test generate_optimized for all versions."""
        response = "Test response"
        query = "Test query"

        for version in PromptVersion:
            prompt = PromptOptimizer.generate_optimized(
                response, query, version
            )

            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # All should contain the response
            assert "response" in prompt.lower()

    def test_prompt_stats(self):
        """Test prompt statistics."""
        prompt = "This is a test prompt with some content"

        stats = PromptOptimizer.get_prompt_stats(prompt)

        assert "token_estimate" in stats
        assert "character_count" in stats
        assert "line_count" in stats
        assert stats["character_count"] == len(prompt)

    def test_prompt_stats_compact_flag(self):
        """Test compact flag in stats."""
        short_prompt = "Brief prompt"
        long_prompt = "x" * 1500  # ~375 tokens

        short_stats = PromptOptimizer.get_prompt_stats(short_prompt)
        long_stats = PromptOptimizer.get_prompt_stats(long_prompt)

        assert short_stats["is_compact"] is True
        assert long_stats["is_compact"] is False

    def test_compare_versions(self):
        """Test comparing all prompt versions."""
        response = "Sample response for testing"
        query = "Sample query"

        comparison = PromptOptimizer.compare_versions(response, query)

        # Should have all versions
        assert len(comparison) == 4
        assert PromptVersion.V1_STANDARD.value in comparison
        assert PromptVersion.V2_COMPACT.value in comparison
        assert PromptVersion.V3_ULTRA_COMPACT.value in comparison
        assert PromptVersion.V4_SELECTIVE.value in comparison

    def test_version_size_progression(self):
        """Test that versions get progressively smaller."""
        response = "This is a test response that contains some content"
        query = "Test query"

        v1 = PromptOptimizer.generate_v1_standard(response, query)
        v2 = PromptOptimizer.generate_v2_compact(response, query)
        v3 = PromptOptimizer.generate_v3_ultra_compact(response, query)

        # Verify size progression
        assert len(v1) > len(v2)
        assert len(v2) > len(v3)

    def test_version_token_progression(self):
        """Test token count progression."""
        response = "Sample response for evaluation testing purposes"
        query = "Sample query for evaluation"

        v1_tokens = PromptOptimizer.estimate_tokens(
            PromptOptimizer.generate_v1_standard(response, query)
        )
        v2_tokens = PromptOptimizer.estimate_tokens(
            PromptOptimizer.generate_v2_compact(response, query)
        )
        v3_tokens = PromptOptimizer.estimate_tokens(
            PromptOptimizer.generate_v3_ultra_compact(response, query)
        )

        # V2 should be <300, V3 should be <200
        assert v2_tokens < 300
        assert v3_tokens < 200
        assert v1_tokens > v2_tokens > v3_tokens

    def test_response_truncation_v2(self):
        """Test response truncation in V2."""
        long_response = "x" * 1000
        query = "Query"

        prompt = PromptOptimizer.generate_v2_compact(long_response, query)

        # Should truncate long response to 500 chars
        assert len(prompt) < 800
        assert len(long_response) > len(prompt)

    def test_response_truncation_v3(self):
        """Test response truncation in V3."""
        long_response = "x" * 1000
        query = "Query"

        prompt = PromptOptimizer.generate_v3_ultra_compact(long_response, query)

        # Should truncate more aggressively to 300 chars
        assert len(prompt) < 500

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        response = "Response"

        for version in PromptVersion:
            prompt = PromptOptimizer.generate_optimized(response, "", version)

            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_special_characters_handling(self):
        """Test handling of special characters."""
        response = 'Response with "quotes" and\nnewlines and special chars: @#$%'
        query = "Query with <tags> and special chars"

        prompt = PromptOptimizer.generate_v2_compact(response, query)

        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestPromptExperiment:
    """Tests for prompt experiment tracking."""

    def test_experiment_initialization(self):
        """Test experiment creation."""
        exp = PromptExperiment(PromptVersion.V2_COMPACT)

        assert exp.version == PromptVersion.V2_COMPACT
        assert len(exp.results) == 0
        assert exp.success_count == 0
        assert exp.failure_count == 0

    def test_experiment_record_success(self):
        """Test recording successful evaluations."""
        exp = PromptExperiment(PromptVersion.V2_COMPACT)

        exp.record_evaluation(
            success=True,
            latency_ms=150.5,
            tokens_used=280,
            quality_score=0.85,
        )

        assert exp.success_count == 1
        assert exp.failure_count == 0
        assert len(exp.results) == 1

    def test_experiment_record_failure(self):
        """Test recording failed evaluations."""
        exp = PromptExperiment(PromptVersion.V3_ULTRA_COMPACT)

        exp.record_evaluation(
            success=False,
            latency_ms=0.0,
            tokens_used=0,
        )

        assert exp.failure_count == 1
        assert exp.success_count == 0

    def test_experiment_mixed_results(self):
        """Test recording mixed success/failure."""
        exp = PromptExperiment(PromptVersion.V2_COMPACT)

        exp.record_evaluation(True, 150.0, 280)
        exp.record_evaluation(True, 160.0, 285)
        exp.record_evaluation(False, 0.0, 0)
        exp.record_evaluation(True, 140.0, 275)

        assert exp.success_count == 3
        assert exp.failure_count == 1
        assert len(exp.results) == 4

    def test_experiment_summary_success(self):
        """Test summary on successful experiments."""
        exp = PromptExperiment(PromptVersion.V2_COMPACT)

        for _ in range(5):
            exp.record_evaluation(True, 150.0, 280)

        summary = exp.get_summary()

        assert summary["total_runs"] == 5
        assert summary["successful"] == 5
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0
        assert 145 < summary["avg_latency_ms"] < 155
        assert 275 < summary["avg_tokens"] < 285

    def test_experiment_summary_partial_success(self):
        """Test summary with mixed results."""
        exp = PromptExperiment(PromptVersion.V3_ULTRA_COMPACT)

        exp.record_evaluation(True, 100.0, 200)
        exp.record_evaluation(True, 120.0, 220)
        exp.record_evaluation(False, 0.0, 0)
        exp.record_evaluation(True, 110.0, 210)

        summary = exp.get_summary()

        assert summary["total_runs"] == 4
        assert summary["successful"] == 3
        assert summary["failed"] == 1
        assert 0.7 < summary["success_rate"] < 0.8
        assert 100 < summary["avg_latency_ms"] < 130
        assert 200 < summary["avg_tokens"] < 220

    def test_experiment_multiple_recordings(self):
        """Test multiple experiment recordings."""
        exp = PromptExperiment(PromptVersion.V2_COMPACT)

        latencies = [150, 160, 140, 155, 145]
        tokens = [280, 285, 275, 290, 278]

        for lat, tok in zip(latencies, tokens):
            exp.record_evaluation(True, lat, tok)

        summary = exp.get_summary()

        # Verify averaging
        expected_avg_latency = sum(latencies) / len(latencies)
        expected_avg_tokens = sum(tokens) / len(tokens)

        assert abs(summary["avg_latency_ms"] - expected_avg_latency) < 1
        assert abs(summary["avg_tokens"] - expected_avg_tokens) < 1

    def test_experiment_version_tracking(self):
        """Test that experiments track their version."""
        exp_v1 = PromptExperiment(PromptVersion.V1_STANDARD)
        exp_v2 = PromptExperiment(PromptVersion.V2_COMPACT)
        exp_v3 = PromptExperiment(PromptVersion.V3_ULTRA_COMPACT)

        assert exp_v1.get_summary()["version"] == PromptVersion.V1_STANDARD.value
        assert exp_v2.get_summary()["version"] == PromptVersion.V2_COMPACT.value
        assert exp_v3.get_summary()["version"] == PromptVersion.V3_ULTRA_COMPACT.value


class TestPromptOptimizationScenarios:
    """Integration tests for prompt optimization scenarios."""

    def test_latency_optimization_workflow(self):
        """Test typical latency optimization workflow."""
        response = "This is a longer response that provides detailed information"
        query = "What is the meaning of life?"

        # Compare all versions
        comparison = PromptOptimizer.compare_versions(response, query)

        # V2 should be <300 tokens
        v2_stats = comparison[PromptVersion.V2_COMPACT.value]["stats"]
        assert v2_stats["is_compact"] is True
        assert v2_stats["token_estimate"] < 300

        # V3 should be <200 tokens
        v3_stats = comparison[PromptVersion.V3_ULTRA_COMPACT.value]["stats"]
        assert v3_stats["token_estimate"] < 200

    def test_prompt_quality_check(self):
        """Test prompt quality validation."""
        response = "Test response"
        query = "Test query"

        v2 = PromptOptimizer.generate_v2_compact(response, query)
        v2_stats = PromptOptimizer.get_prompt_stats(v2)

        # Verify quality: must contain JSON structure
        assert "{" in v2
        assert "}" in v2
        # Must contain key dimensions
        assert any(dim in v2 for dim in ["accuracy", "relevance", "safety"])
