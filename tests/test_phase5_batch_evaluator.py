"""
Phase 5 Day 2: Batch Evaluator Tests

Tests for batch evaluation capability:
  - Concurrent evaluation
  - Error handling and retry logic
  - Timeout management
  - Progress tracking
  - Deduplication
  - Performance under load
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from src.guardrails.batch_evaluator import (
    BatchEvaluationItem,
    BatchEvaluationResult,
    BatchEvaluator,
    SmartBatchEvaluator,
)


class TestBatchEvaluationItem:
    """Tests for batch item dataclass."""

    def test_item_initialization(self):
        """Test batch item creation."""
        item = BatchEvaluationItem(
            item_id="item_1",
            response="Response text",
            query="Query text",
            agent_id="agent_1",
        )

        assert item.item_id == "item_1"
        assert item.response == "Response text"
        assert item.query == "Query text"
        assert item.agent_id == "agent_1"

    def test_item_default_metadata(self):
        """Test default metadata initialization."""
        item = BatchEvaluationItem(
            item_id="item_1",
            response="Response",
        )

        assert item.metadata == {}


class TestBatchEvaluationResult:
    """Tests for batch result dataclass."""

    def test_result_initialization_success(self):
        """Test successful result creation."""
        eval_dict = {"accuracy": 0.85}
        result = BatchEvaluationResult(
            item_id="item_1",
            success=True,
            evaluation=eval_dict,
            elapsed_ms=150.5,
            from_cache=False,
        )

        assert result.item_id == "item_1"
        assert result.success is True
        assert result.evaluation == eval_dict
        assert result.elapsed_ms == 150.5
        assert result.error is None

    def test_result_initialization_failure(self):
        """Test failure result creation."""
        result = BatchEvaluationResult(
            item_id="item_2",
            success=False,
            error="Evaluation timeout",
            elapsed_ms=30000.0,
        )

        assert result.success is False
        assert result.error == "Evaluation timeout"
        assert result.evaluation is None


class TestBatchEvaluator:
    """Tests for batch evaluator."""

    @pytest.mark.asyncio
    async def test_evaluator_initialization(self):
        """Test evaluator creation."""
        async_eval_fn = AsyncMock(return_value={"accuracy": 0.85})

        evaluator = BatchEvaluator(
            evaluate_fn=async_eval_fn,
            max_concurrent=5,
            timeout_seconds=10,
            max_retries=2,
        )

        assert evaluator.max_concurrent == 5
        assert evaluator.timeout_seconds == 10
        assert evaluator.max_retries == 2

    @pytest.mark.asyncio
    async def test_evaluate_single_item_success(self):
        """Test evaluating single item."""
        async_eval_fn = AsyncMock(return_value={"accuracy": 0.85})

        evaluator = BatchEvaluator(evaluate_fn=async_eval_fn, max_concurrent=1)

        item = BatchEvaluationItem(
            item_id="item_1",
            response="Response",
            query="Query",
        )

        results, stats = await evaluator.evaluate_batch([item])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].evaluation == {"accuracy": 0.85}
        assert stats["successful"] == 1
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_multiple_items(self):
        """Test evaluating multiple items."""
        counter = 0

        async def mock_eval(response, query, agent_id):
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.01)
            return {"accuracy": 0.80 + counter * 0.01}

        evaluator = BatchEvaluator(evaluate_fn=mock_eval, max_concurrent=3)

        items = [
            BatchEvaluationItem(
                item_id=f"item_{i}",
                response=f"Response {i}",
                query=f"Query {i}",
            )
            for i in range(5)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        assert len(results) == 5
        assert stats["successful"] == 5
        assert stats["success_rate"] == 1.0
        assert counter == 5

    @pytest.mark.asyncio
    async def test_evaluate_with_failure(self):
        """Test batch with failed items."""
        call_count = 0

        async def mock_eval_with_failure(response, query, agent_id):
            nonlocal call_count
            call_count += 1
            if "fail" in response:
                raise ValueError("Simulated failure")
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=mock_eval_with_failure,
            max_concurrent=2,
            timeout_seconds=1,
            max_retries=1,
        )

        items = [
            BatchEvaluationItem(item_id="item_1", response="Normal response"),
            BatchEvaluationItem(item_id="item_2", response="This will fail"),
        ]

        results, stats = await evaluator.evaluate_batch(items)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert stats["successful"] == 1
        assert stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_timeout(self):
        """Test timeout handling."""

        async def slow_eval(response, query, agent_id):
            await asyncio.sleep(5)  # Very slow
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=slow_eval,
            max_concurrent=1,
            timeout_seconds=0.1,
            max_retries=1,
        )

        item = BatchEvaluationItem(item_id="item_1", response="Response")

        results, stats = await evaluator.evaluate_batch([item])

        assert results[0].success is False
        assert "Timeout" in results[0].error

    @pytest.mark.asyncio
    async def test_concurrency_control(self):
        """Test max_concurrent limit."""
        concurrent_count = 0
        max_concurrent_observed = 0

        async def tracked_eval(response, query, agent_id):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

            await asyncio.sleep(0.05)

            concurrent_count -= 1
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(evaluate_fn=tracked_eval, max_concurrent=2)

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(10)
        ]

        await evaluator.evaluate_batch(items)

        # Should never exceed max_concurrent
        assert max_concurrent_observed <= 2

    @pytest.mark.asyncio
    async def test_batch_statistics(self):
        """Test batch statistics tracking."""

        async def mock_eval(response, query, agent_id):
            await asyncio.sleep(0.01)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(evaluate_fn=mock_eval, max_concurrent=3)

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(5)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        assert stats["total_items"] == 5
        assert stats["successful"] == 5
        assert stats["failed"] == 0
        assert 0 < stats["avg_latency_ms"]

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry on transient failure."""
        call_count = 0

        async def flaky_eval(response, query, agent_id):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise ValueError("Transient error")
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=flaky_eval,
            max_concurrent=1,
            max_retries=3,
        )

        item = BatchEvaluationItem(item_id="item_1", response="Response")

        results, stats = await evaluator.evaluate_batch([item])

        # Should succeed after retry
        assert results[0].success is True
        assert call_count == 2  # Initial + 1 retry


class TestSmartBatchEvaluator:
    """Tests for smart batch evaluator with deduplication."""

    @pytest.mark.asyncio
    async def test_smart_batch_deduplication(self):
        """Test deduplication of identical responses."""
        call_count = 0

        async def counting_eval(response, query, agent_id):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {"accuracy": 0.85}

        evaluator = SmartBatchEvaluator(
            evaluate_fn=counting_eval,
            max_concurrent=3,
        )

        # Create items with duplicate responses
        items = [
            BatchEvaluationItem(item_id="item_1", response="Same response"),
            BatchEvaluationItem(item_id="item_2", response="Same response"),
            BatchEvaluationItem(item_id="item_3", response="Same response"),
            BatchEvaluationItem(item_id="item_4", response="Different response"),
        ]

        results, stats = await evaluator.evaluate_batch_smart(items)

        assert len(results) == 4
        # Should only evaluate 2 unique responses
        assert stats["unique_responses"] == 2
        assert stats["deduplication_savings"] == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_smart_batch_result_mapping(self):
        """Test result mapping in smart batch."""

        async def mock_eval(response, query, agent_id):
            return {
                "accuracy": 0.80 if "good" in response.lower() else 0.50,
                "relevance": 0.90,
            }

        evaluator = SmartBatchEvaluator(evaluate_fn=mock_eval, max_concurrent=3)

        items = [
            BatchEvaluationItem(item_id="item_1", response="good response"),
            BatchEvaluationItem(item_id="item_2", response="good response"),
            BatchEvaluationItem(item_id="item_3", response="bad response"),
        ]

        results, stats = await evaluator.evaluate_batch_smart(items)

        # All items with same response should have same evaluation
        assert results[0].evaluation == results[1].evaluation
        # Items with different responses should have different evaluations
        assert results[0].evaluation["accuracy"] == 0.80
        assert results[2].evaluation["accuracy"] == 0.50

    @pytest.mark.asyncio
    async def test_smart_batch_no_duplicates(self):
        """Test smart batch with no duplicates."""
        call_count = 0

        async def counting_eval(response, query, agent_id):
            nonlocal call_count
            call_count += 1
            return {"accuracy": 0.85}

        evaluator = SmartBatchEvaluator(
            evaluate_fn=counting_eval,
            max_concurrent=5,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(5)
        ]

        results, stats = await evaluator.evaluate_batch_smart(items)

        # All unique, should evaluate all 5
        assert len(results) == 5
        assert stats["unique_responses"] == 5
        assert stats["deduplication_savings"] == 0
        assert call_count == 5


class TestBatchPerformance:
    """Performance tests for batch evaluator."""

    @pytest.mark.asyncio
    async def test_batch_throughput(self):
        """Test batch evaluation throughput."""

        async def fast_eval(response, query, agent_id):
            await asyncio.sleep(0.001)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(evaluate_fn=fast_eval, max_concurrent=10)

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(100)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        # Should complete all 100 items
        assert stats["successful"] == 100
        # With 10 concurrent and 1ms each, should be roughly 10ms total
        # Allow generous margin
        assert stats["total_elapsed_ms"] < 5000

    @pytest.mark.asyncio
    async def test_batch_error_resilience(self):
        """Test batch continues despite errors."""
        failures = ["item_3", "item_7"]

        async def sometimes_fails(response, query, agent_id):
            if query in failures:
                raise ValueError("Simulated error")
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=sometimes_fails,
            max_concurrent=5,
            max_retries=1,
        )

        items = [
            BatchEvaluationItem(
                item_id=f"item_{i}",
                response=f"Response {i}",
                query=f"item_{i}",
            )
            for i in range(10)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        # Should complete all items
        assert len(results) == 10
        # Exactly 2 should fail
        assert stats["failed"] == 2
        assert stats["successful"] == 8


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
