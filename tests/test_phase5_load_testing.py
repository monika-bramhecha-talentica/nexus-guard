"""
Phase 5 Day 2: Load Testing with Concurrent Evaluations

Comprehensive load tests targeting 50+ concurrent evaluations:
  - Small load (50 concurrent items)
  - Medium load (100 concurrent items)
  - Heavy load (200+ concurrent items)
  - Sustained load (1000+ items over time)
  - Error handling under load
  - Cache effectiveness under load
  - Smart batch deduplication under load
  - Latency tail behavior under high concurrency
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import AsyncMock

from src.guardrails.batch_evaluator import BatchEvaluationItem, BatchEvaluator, SmartBatchEvaluator
from src.guardrails.evaluation_cache import LayeredEvaluationCache
from src.guardrails.prompt_optimizer import PromptOptimizer, PromptVersion


class TestSmallLoad:
    """Load tests with 50 concurrent items."""

    @pytest.mark.asyncio
    async def test_50_concurrent_items(self):
        """Test 50 concurrent evaluations."""
        async def mock_eval(response, query, agent_id):
            await asyncio.sleep(0.01)
            return {"accuracy": 0.85, "relevance": 0.90}

        evaluator = BatchEvaluator(
            evaluate_fn=mock_eval,
            max_concurrent=10,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(50)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        print(f"\n50 Items Load Test - Elapsed: {elapsed:.2f}s, Throughput: {50/elapsed:.1f} items/sec")

        assert len(results) == 50
        assert stats["successful"] == 50
        assert stats["success_rate"] == 1.0
        # Should complete reasonably quickly (50 items * 10ms each, with 10 concurrent = ~50ms)
        assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_50_concurrent_with_mixed_latency(self):
        """Test 50 items with variable latency."""
        async def variable_eval(response, query, agent_id):
            # Mix of fast (5ms) and normal (20ms) items
            delay = 0.005 if int(response.split("_")[1]) % 3 == 0 else 0.02
            await asyncio.sleep(delay)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=variable_eval,
            max_concurrent=10,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i}")
            for i in range(50)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        assert stats["successful"] == 50
        # Should handle variable latencies well
        assert stats["avg_latency_ms"] > 0


class TestMediumLoad:
    """Load tests with 100 concurrent items."""

    @pytest.mark.asyncio
    async def test_100_concurrent_items(self):
        """Test 100 concurrent evaluations."""
        async def mock_eval(response, query, agent_id):
            await asyncio.sleep(0.005)
            return {"accuracy": 0.85, "relevance": 0.90, "safety": 0.95}

        evaluator = BatchEvaluator(
            evaluate_fn=mock_eval,
            max_concurrent=20,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(100)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        print(f"\n100 Items Load Test - Elapsed: {elapsed:.2f}s, Throughput: {100/elapsed:.1f} items/sec")

        assert len(results) == 100
        assert stats["successful"] == 100
        # 100 items * 5ms each, with 20 concurrent = ~25ms
        assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_100_concurrent_with_retries(self):
        """Test 100 items with occasional failures requiring retries."""
        call_count = 0

        async def flaky_eval(response, query, agent_id):
            nonlocal call_count
            call_count += 1

            # 10% failure rate (will retry)
            if call_count % 10 == 0:
                raise ValueError("Simulated transient error")

            await asyncio.sleep(0.005)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=flaky_eval,
            max_concurrent=15,
            max_retries=2,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(100)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        # Should succeed for most despite transient failures
        assert stats["successful"] > 90
        assert len(results) == 100


class TestHeavyLoad:
    """Load tests with 200+ concurrent items."""

    @pytest.mark.asyncio
    async def test_200_concurrent_items(self):
        """Test 200 concurrent evaluations."""
        async def fast_eval(response, query, agent_id):
            await asyncio.sleep(0.002)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=fast_eval,
            max_concurrent=30,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(200)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        print(f"\n200 Items Load Test - Elapsed: {elapsed:.2f}s, Throughput: {200/elapsed:.1f} items/sec")

        assert len(results) == 200
        assert stats["successful"] == 200
        assert elapsed < 30.0

    @pytest.mark.asyncio
    async def test_500_concurrent_items(self):
        """Test 500 concurrent evaluations."""
        async def ultra_fast_eval(response, query, agent_id):
            await asyncio.sleep(0.001)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=ultra_fast_eval,
            max_concurrent=50,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(500)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        print(f"\n500 Items Load Test - Elapsed: {elapsed:.2f}s, Throughput: {500/elapsed:.1f} items/sec")

        assert len(results) == 500
        assert stats["successful"] == 500
        # Should handle 500 items efficiently
        assert elapsed < 60.0


class TestSustainedLoad:
    """Sustained load tests with large item counts."""

    @pytest.mark.asyncio
    async def test_1000_items_sustained(self):
        """Test 1000 items in batches."""
        async def mock_eval(response, query, agent_id):
            await asyncio.sleep(0.001)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=mock_eval,
            max_concurrent=50,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(1000)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        print(f"\n1000 Items Sustained Test - Elapsed: {elapsed:.2f}s, Throughput: {1000/elapsed:.1f} items/sec")

        assert len(results) == 1000
        assert stats["successful"] == 1000
        assert stats["success_rate"] == 1.0


class TestCacheEffectivenessUnderLoad:
    """Test cache hit rates under high load."""

    @pytest.mark.asyncio
    async def test_cache_effectiveness_50_concurrent(self):
        """Test cache with 50 concurrent items, many duplicates."""
        cache = LayeredEvaluationCache(use_redis=False)
        call_count = 0

        async def cached_eval(response, query, agent_id):
            nonlocal call_count

            cache_key = f"eval_{response}"
            cached = cache.get(cache_key)
            if cached:
                return cached

            call_count += 1
            await asyncio.sleep(0.01)
            result = {"accuracy": 0.85, "id": call_count}
            cache.set(cache_key, result)
            return result

        evaluator = BatchEvaluator(
            evaluate_fn=cached_eval,
            max_concurrent=10,
        )

        # Create 50 items with only 10 unique responses
        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i % 10}")
            for i in range(50)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        print(f"\nCache Effectiveness - Actual evaluations: {call_count}/50")

        assert len(results) == 50
        # With cache, should only evaluate ~10 unique responses (plus some overhead from concurrency)
        assert call_count <= 15

    @pytest.mark.asyncio
    async def test_smart_batch_deduplication_large(self):
        """Test smart batch with large duplicate set."""
        call_count = 0

        async def counting_eval(response, query, agent_id):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.005)
            return {"accuracy": 0.85}

        evaluator = SmartBatchEvaluator(
            evaluate_fn=counting_eval,
            max_concurrent=20,
        )

        # 100 items with only 20 unique responses
        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i % 20}")
            for i in range(100)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch_smart(items)
        elapsed = time.perf_counter() - start

        print(f"\nSmart Batch Deduplication - Evaluated {stats['unique_responses']} unique, saved {stats['deduplication_savings']} evaluations")

        assert len(results) == 100
        assert stats["unique_responses"] == 20
        assert stats["deduplication_savings"] == 80
        assert call_count == 20


class TestErrorResilienceUnderLoad:
    """Test error handling and recovery under load."""

    @pytest.mark.asyncio
    async def test_partial_failures_100_items(self):
        """Test 100 items with 20% failure rate."""
        async def sometimes_fails(response, query, agent_id):
            # 20% fail
            if int(response.split("_")[1]) % 5 == 0:
                raise ValueError("Simulated error")

            await asyncio.sleep(0.005)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=sometimes_fails,
            max_concurrent=20,
            max_retries=1,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i}")
            for i in range(100)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        print(f"\nError Resilience - Success: {stats['successful']}, Failed: {stats['failed']}")

        assert len(results) == 100
        # Should succeed for 80% (indices not divisible by 5)
        assert stats["successful"] == 80
        assert stats["failed"] == 20

    @pytest.mark.asyncio
    async def test_timeout_resilience_200_items(self):
        """Test timeout handling with 200 items."""
        async def variable_speed(response, query, agent_id):
            # 5% are slow
            if int(response.split("_")[1]) % 20 == 0:
                await asyncio.sleep(5)  # Will timeout
            else:
                await asyncio.sleep(0.005)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=variable_speed,
            max_concurrent=25,
            timeout_seconds=0.1,
            max_retries=1,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i}")
            for i in range(200)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        print(f"\nTimeout Resilience - Completed {stats['successful']}/{len(results)} items")

        assert len(results) == 200
        # Should timeout on ~10 items (200 / 20), but retry and eventually fail
        assert stats["failed"] < 20  # Some may still succeed on retry
        assert stats["successful"] > 180


class TestLatencyTailBehavior:
    """Test P50/P95/P99 latencies under various loads."""

    @pytest.mark.asyncio
    async def test_latency_percentiles_100_concurrent(self):
        """Measure latency distribution with 100 items."""
        async def eval_with_jitter(response, query, agent_id):
            # Add realistic jitter based on item hash
            base_delay = 0.005
            jitter = (hash(response) % 3) * 0.002 / 3000
            await asyncio.sleep(base_delay + jitter)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=eval_with_jitter,
            max_concurrent=20,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i}")
            for i in range(100)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        latencies = [r.elapsed_ms for r in results]
        sorted_lat = sorted(latencies)

        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        print(f"\nLatency Percentiles (100 items) - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")

        assert stats["successful"] == 100
        # Latencies should follow reasonable distribution
        assert p50 < 100
        assert p95 > p50

    @pytest.mark.asyncio
    async def test_latency_under_max_concurrency(self):
        """Test latency when hitting max concurrency limits."""
        latencies = []
        lock = asyncio.Lock()

        async def tracked_eval(response, query, agent_id):
            start = time.perf_counter()
            await asyncio.sleep(0.01)
            elapsed = (time.perf_counter() - start) * 1000

            async with lock:
                latencies.append(elapsed)

            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=tracked_eval,
            max_concurrent=50,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response_{i}")
            for i in range(200)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        sorted_lat = sorted(latencies)
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        print(f"\nP99 Latency at Max Concurrency: {p99:.1f}ms")

        assert stats["successful"] == 200


class TestPromptOptimizationUnderLoad:
    """Test prompt optimization effectiveness with multiple versions under load."""

    def test_prompt_generation_speed_under_load(self):
        """Test prompt generation speed for 1000 items."""
        response = "This is a test response for evaluation purposes"
        query = "What is the quality?"

        versions = [
            PromptVersion.V1_STANDARD,
            PromptVersion.V2_COMPACT,
            PromptVersion.V3_ULTRA_COMPACT,
        ]

        for version in versions:
            latencies = []
            for _ in range(1000):
                start = time.perf_counter()
                prompt = PromptOptimizer.generate_optimized(response, query, version)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            avg = statistics.mean(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]

            print(f"\n{version.value} - 1000 generations: Avg {avg:.3f}ms, P95: {p95:.3f}ms")

            # All versions should be very fast (< 1ms average)
            assert avg < 1.0


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
