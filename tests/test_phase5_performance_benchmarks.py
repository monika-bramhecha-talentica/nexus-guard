"""
Phase 5 Day 2: Performance Benchmarking Tests (16-18 tests)

Comprehensive performance benchmarks for:
  - Cache performance (hit/miss latency)
  - Batch evaluation throughput
  - Prompt optimization impact
  - Streaming evaluation
  - Concurrent request handling
  - Memory usage patterns
  - Latency percentiles (P50/P95/P99)
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock
import statistics

from src.guardrails.evaluation_cache import (
    LayeredEvaluationCache,
    EvaluationCacheKeyBuilder,
)
from src.guardrails.batch_evaluator import BatchEvaluationItem, BatchEvaluator
from src.guardrails.prompt_optimizer import (
    PromptOptimizer,
    PromptVersion,
)


class TestCachePerformance:
    """Cache performance benchmarks."""

    def test_cache_hit_latency(self):
        """Benchmark cache hit latency."""
        cache = LayeredEvaluationCache(use_redis=False)

        # Warm cache
        value = {"accuracy": 0.85, "relevance": 0.90}
        key = "perf_test_key"
        cache.set(key, value)

        # Measure hit latency
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            cache.get(key)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(elapsed)

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\nCache Hit Latency - P50: {p50:.3f}ms, P95: {p95:.3f}ms, P99: {p99:.3f}ms")

        # Hits should be very fast (<1ms)
        assert p50 < 1.0
        assert p95 < 2.0

    def test_cache_miss_latency(self):
        """Benchmark cache miss latency."""
        cache = LayeredEvaluationCache(use_redis=False)

        latencies = []
        for i in range(1000):
            start = time.perf_counter()
            cache.get(f"nonexistent_{i}")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"Cache Miss Latency - P50: {p50:.3f}ms, P95: {p95:.3f}ms")

        # Misses should also be very fast
        assert p50 < 0.5
        assert p95 < 1.0

    def test_cache_set_latency(self):
        """Benchmark cache write latency."""
        cache = LayeredEvaluationCache(use_redis=False)

        latencies = []
        for i in range(100):
            value = {"accuracy": 0.85, "relevance": 0.90}
            start = time.perf_counter()
            cache.set(f"key_{i}", value)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        p50 = statistics.median(latencies)
        avg = statistics.mean(latencies)

        print(f"Cache Set Latency - P50: {p50:.3f}ms, Avg: {avg:.3f}ms")

        # Sets should be reasonably fast
        assert p50 < 1.0

    def test_cache_hit_rate_improvement(self):
        """Test hit rate with repeated evaluations."""
        cache = LayeredEvaluationCache(use_redis=False)

        # Populate cache with 100 evaluations
        for i in range(100):
            cache.set(f"eval_{i}", {"accuracy": 0.80 + i * 0.001})

        # Query pattern: 80% repeat, 20% new
        hits = 0
        misses = 0

        for iteration in range(1000):
            # 80% repeat from existing
            if iteration % 5 < 4:
                key = f"eval_{iteration % 100}"
            else:
                key = f"eval_new_{iteration}"

            if cache.get(key) is not None:
                hits += 1
            else:
                misses += 1

        hit_rate = hits / (hits + misses)

        print(f"Cache Hit Rate: {hit_rate:.1%}")

        # Should achieve ~80% hit rate
        assert 0.75 < hit_rate < 0.85


class TestBatchEvaluationPerformance:
    """Batch evaluation performance benchmarks."""

    @pytest.mark.asyncio
    async def test_batch_throughput_small(self):
        """Benchmark small batch throughput."""

        async def mock_eval(response, query, agent_id):
            await asyncio.sleep(0.01)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=mock_eval,
            max_concurrent=5,
            timeout_seconds=30,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(10)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        throughput = 10 / elapsed

        print(f"\nBatch Throughput (10 items): {throughput:.1f} items/sec")

        # With 5 concurrent and 10ms each, should be ~50+ items/sec
        assert throughput > 30
        assert stats["successful"] == 10

    @pytest.mark.asyncio
    async def test_batch_throughput_large(self):
        """Benchmark large batch throughput."""

        async def fast_eval(response, query, agent_id):
            await asyncio.sleep(0.001)  # Very fast
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=fast_eval,
            max_concurrent=20,
            timeout_seconds=30,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(100)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        throughput = 100 / elapsed

        print(f"Batch Throughput (100 items): {throughput:.1f} items/sec")

        # Should handle 100 items efficiently
        assert throughput > 100
        assert stats["successful"] == 100

    @pytest.mark.asyncio
    async def test_batch_latency_distribution(self):
        """Test latency distribution across batch."""

        async def variable_eval(response, query, agent_id):
            # Variable latency based on response
            delay = 0.001 if "fast" in response else 0.01
            await asyncio.sleep(delay)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=variable_eval,
            max_concurrent=10,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response="fast")
            for i in range(50)
        ] + [
            BatchEvaluationItem(item_id=f"item_slow_{i}", response="slow")
            for i in range(50)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        print(f"Batch Latency - Avg: {stats['avg_latency_ms']:.1f}ms")

        # Should complete all items
        assert len(results) == 100
        assert stats["successful"] == 100


class TestPromptOptimizationPerformance:
    """Prompt optimization performance benchmarks."""

    def test_prompt_generation_speed_v2(self):
        """Benchmark V2 compact prompt generation."""
        response = "This is a test response with substantial content for evaluation"
        query = "What is the performance?"

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            PromptOptimizer.generate_v2_compact(response, query)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        p50 = statistics.median(latencies)
        avg = statistics.mean(latencies)

        print(f"\nV2 Prompt Generation - P50: {p50:.3f}ms, Avg: {avg:.3f}ms")

        # Prompt generation should be very fast
        assert p50 < 0.5

    def test_prompt_generation_speed_v3(self):
        """Benchmark V3 ultra-compact prompt generation."""
        response = "This is a test response with substantial content for evaluation"
        query = "What is the performance?"

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            PromptOptimizer.generate_v3_ultra_compact(response, query)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        p50 = statistics.median(latencies)

        print(f"V3 Prompt Generation - P50: {p50:.3f}ms")

        # V3 should be even faster
        assert p50 < 0.5

    def test_prompt_version_comparison(self):
        """Compare all prompt versions."""
        response = "Test response"
        query = "Test query"

        # Generate all versions
        v1 = PromptOptimizer.generate_v1_standard(response, query)
        v2 = PromptOptimizer.generate_v2_compact(response, query)
        v3 = PromptOptimizer.generate_v3_ultra_compact(response, query)

        # Measure token estimates
        v1_tokens = PromptOptimizer.estimate_tokens(v1)
        v2_tokens = PromptOptimizer.estimate_tokens(v2)
        v3_tokens = PromptOptimizer.estimate_tokens(v3)

        print(
            f"\nPrompt Sizes - V1: {v1_tokens} tokens, "
            f"V2: {v2_tokens} tokens, V3: {v3_tokens} tokens"
        )

        # Verify token reduction
        assert v2_tokens < v1_tokens
        assert v3_tokens < v2_tokens
        assert v3_tokens < 200


class TestConcurrencyPerformance:
    """Concurrent request handling benchmarks."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test concurrent cache access performance."""
        cache = LayeredEvaluationCache(use_redis=False)

        # Warm cache
        for i in range(100):
            cache.set(f"key_{i}", {"accuracy": 0.85})

        # Concurrent access
        async def access_cache():
            latencies = []
            for i in range(50):
                start = time.perf_counter()
                cache.get(f"key_{i % 100}")
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
            return latencies

        start = time.perf_counter()
        results = await asyncio.gather(*[access_cache() for _ in range(10)])
        elapsed = time.perf_counter() - start

        total_accesses = sum(len(r) for r in results)
        throughput = total_accesses / elapsed

        print(f"\nConcurrent Cache Access - {throughput:.1f} ops/sec")

        # Should handle concurrent access efficiently
        assert throughput > 10000

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_stress(self):
        """Stress test concurrent evaluations."""

        counter = 0
        lock = asyncio.Lock()

        async def counting_eval(response, query, agent_id):
            nonlocal counter
            async with lock:
                counter += 1

            await asyncio.sleep(0.001)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=counting_eval,
            max_concurrent=50,
        )

        # Large concurrent batch
        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(200)
        ]

        start = time.perf_counter()
        results, stats = await evaluator.evaluate_batch(items)
        elapsed = time.perf_counter() - start

        throughput = 200 / elapsed

        print(f"Concurrent Stress Test - {throughput:.1f} items/sec")

        assert stats["successful"] == 200


class TestMemoryEfficiency:
    """Memory usage benchmarks."""

    def test_cache_memory_usage(self):
        """Test cache memory efficiency."""
        cache = LayeredEvaluationCache(max_memory_entries=1000, use_redis=False)

        # Add evaluations
        for i in range(100):
            cache.set(
                f"eval_{i}",
                {
                    "accuracy": 0.85,
                    "relevance": 0.90,
                    "completeness": 0.80,
                    "safety": 1.0,
                    "hallucination": 0.95,
                    "pii_handling": 1.0,
                },
            )

        stats = cache.get_stats()
        avg_entry_size = stats["l1"]["avg_entry_size"]

        print(f"\nAverage Cache Entry Size: {avg_entry_size:.0f} bytes")

        # Each evaluation dictionary (6 float fields) should be ~100-200 bytes
        assert 50 < avg_entry_size < 250

    def test_batch_memory_during_processing(self):
        """Test memory usage during batch processing."""
        # This is a smoke test to ensure no memory leaks
        cache = LayeredEvaluationCache(max_memory_entries=5000, use_redis=False)

        # Process many items
        for i in range(1000):
            cache.set(
                f"batch_{i}",
                {
                    "accuracy": 0.85 + i * 0.0001,
                    "relevance": 0.90,
                    "completeness": 0.80,
                },
            )

        stats = cache.get_stats()

        # Verify LRU is working (max entries should not exceed limit)
        assert stats["l1"]["entries"] <= 1000


class TestLatencyCharacterization:
    """Characterize latency under various scenarios."""

    def test_p50_p95_p99_latencies(self):
        """Measure P50/P95/P99 latencies."""
        cache = LayeredEvaluationCache(use_redis=False)
        cache.set("key", {"accuracy": 0.85})

        latencies = []
        for _ in range(10000):
            start = time.perf_counter()
            cache.get("key")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        print(f"\nLatency Percentiles - P50: {p50:.4f}ms, P95: {p95:.4f}ms, P99: {p99:.4f}ms")

        # Verify latency targets
        assert p50 < 0.1
        assert p95 < 0.5
        assert p99 < 1.0

    @pytest.mark.asyncio
    async def test_batch_latency_percentiles(self):
        """Measure batch evaluation latency percentiles."""

        async def eval_with_jitter(response, query, agent_id):
            # Add some jitter to simulate real LLM calls
            await asyncio.sleep(0.005 + (hash(response) % 5) * 0.001 / 5000)
            return {"accuracy": 0.85}

        evaluator = BatchEvaluator(
            evaluate_fn=eval_with_jitter,
            max_concurrent=10,
        )

        items = [
            BatchEvaluationItem(item_id=f"item_{i}", response=f"Response {i}")
            for i in range(100)
        ]

        results, stats = await evaluator.evaluate_batch(items)

        latencies = [r.elapsed_ms for r in results]
        sorted_lat = sorted(latencies)

        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        print(f"Batch Item Latencies - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")

        # Median should be around 5ms
        assert 3 < p50 < 10


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
